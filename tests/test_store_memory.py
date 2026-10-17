"""A store that did not deliver to a family's place (or offered no cash on delivery there) is not looked up again for a
while; a "does not deliver" from a store that set another pincode is not believed (live 17 Oct: Zepto set Patna,
Chhattisgarh 497331 for a Patna, Bihar home)."""

from datetime import timedelta

from app.core import clock
from app.tasks import runtime
from app.tasks.models import Task


def _task(service, code, pin, *, days=1, metrics=False):
    now = clock.now()
    details = {"limits": {"place": {"pincode": pin}}}
    if metrics:
        details["metrics"] = {"failure_code": code}
    else:
        details["fail_code"] = code
    return Task(family_id="fam-sm", subject_id="p", requested_by="p", service=service, kind="order", goal="eggs", details=details,
                status="failed", phase="browse", created_at=now - timedelta(days=days), updated_at=now - timedelta(days=days),
                deadline_at=now)


async def test_known_unserviceable_and_no_cod(db, at):
    at("2026-10-17 12:00")
    db.add_all([_task("zepto", "not_deliverable", "492001"), _task("instamart", "no_cod", "492001", metrics=True),
                _task("blinkit", "not_deliverable", "800001"), _task("zepto", "no_cod", "492001", days=10)])
    await db.flush()
    known = await runtime.skip_known(db, "fam-sm", {"pincode": "492001"}, ("blinkit", "instamart", "zepto"))
    assert known == {"zepto": "does not deliver there", "instamart": "no cash on delivery there"}
    assert await runtime.skip_known(db, "fam-sm", {}, ("zepto",)) == {}


def test_a_no_from_another_pincode_is_not_believed():
    place = {"pincode": "800001", "nickname": "Home"}
    assert not runtime._location_matches(place, {"location_set": "Patna, Chhattisgarh 497331"})
    assert runtime._location_matches(place, {"location_set": "Aadarsh Chowk, Patna 800001"})
    assert runtime._location_matches(place, {"location_set": "Patna"})
