"""Check-in on the local desk, and the live page check against a stand-in fetch. No Browser Use and no airline is contacted."""

import httpx

from app.tasks import checkin_desk, checkin_pages


def test_checkin_opens_on_the_form_not_the_homepage():
    from app.tasks.checkin_pages import PAGES
    from app.tasks.skills import SKILLS

    for name, url in PAGES.items():
        assert SKILLS[name]["start_url"] == url
        assert url.rstrip("/").count("/") >= 3


def test_every_airline_desk_shows_the_booking_form():
    desk = checkin_desk.Desk()
    base = desk.start()
    try:
        for airline in checkin_desk.AIRLINES:
            form = httpx.get(f"{base}/{airline}/check-in", timeout=5).text
            assert checkin_desk.fields_present(form)
    finally:
        desk.stop()


async def test_a_free_seat_is_taken_and_nothing_is_paid():
    desk = checkin_desk.Desk()
    base = desk.start()
    try:
        out = await checkin_desk.walk(base)
    finally:
        desk.stop()
    assert out["ok"] is True and out["seat"] == "1A" and out["paid"] is False and out["boarding_pass"] is True
    assert "pay" not in " ".join(out["calls"]).lower()


async def test_a_missing_booking_is_not_found_and_a_paid_map_stops():
    desk = checkin_desk.Desk()
    base = desk.start()
    try:
        missing = await checkin_desk.walk(base, pnr="NOBODY", last_name="Devi")
        paid = await checkin_desk.walk(base, mode="paidonly")
    finally:
        desk.stop()
    assert missing["pnr_found"] is False
    assert paid["free_seat"] is False and paid["paid"] is False and "POST /seat" not in paid["calls"]


async def test_the_page_check_keeps_a_form_only_when_the_page_shows_it():
    pages = {
        "indigo": (200, "Web check-in. Booking reference. Last name."),
        "airindia": (200, "Welcome"),
        "airindiaexpress": (200, "Record locator and last name"),
        "akasa": (500, "Booking reference Last name"),
        "spicejet": (200, ""),
    }

    async def fetch(url: str):
        name = next(k for k, u in checkin_pages.PAGES.items() if u == url)
        if name == "spicejet":
            raise TimeoutError("timed out")
        return pages[name]

    got = await checkin_pages.check(fetch)
    assert got["indigo"]["fields"] is True and got["airindia"]["fields"] is False
    assert got["airindiaexpress"]["fields"] is True and got["akasa"]["loaded"] is False
    assert got["spicejet"]["loaded"] is False and got["spicejet"]["error"] == "TimeoutError"


async def test_a_booking_is_read_and_a_missing_one_is_not_found():
    desk = checkin_desk.Desk()
    base = desk.start()
    try:
        got = await checkin_desk.read_booking(base)
        missing = await checkin_desk.read_booking(base, pnr="NOBODY", last_name="Devi")
        rejected = await checkin_desk.read_booking(base, pnr="TESTPN", last_name="Nobody")
        blocked = await checkin_desk.read_booking(base, pnr="BLOCKED", last_name="Devi")
        status = await checkin_desk.read_status(base)
        delayed = await checkin_desk.read_status(base, state="delayed")
        landed = await checkin_desk.read_status(base, state="landed")
        unknown = await checkin_desk.read_status(base, flight="6E-999")
    finally:
        desk.stop()
    assert got["found"] is True and got["flight"] == "6E-204" and got["departs"] == "09:10" and got["arrives"] == "11:05"
    assert got["from_terminal"] == "1" and got["to_terminal"] == "2" and got["status"] == "on time"
    assert got["passengers"] == ["Kamla Devi"] and got["seat"] == "12A" and got["changed"] is False and got["paid"] is False
    assert "pay" not in " ".join(got["calls"]).lower() and "cancel" not in " ".join(got["calls"]).lower()
    assert missing["found"] is False and missing["pnr_found"] is False and missing["changed"] is False
    assert rejected["name_rejected"] is True and rejected["found"] is False and rejected["pnr_found"] is True
    assert blocked["blocked"] is True and blocked["found"] is False and blocked["paid"] is False
    assert status["found"] is True and status["passengers"] == [] and status["seat"] is None and status["flight"] == "6E-204"
    assert delayed["status"] == "delayed" and delayed["arrives"] == "11:40"
    assert landed["status"] == "landed" and landed["found"] is True
    assert unknown["found"] is False
