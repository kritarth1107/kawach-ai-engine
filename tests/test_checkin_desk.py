"""Check-in on the local desk, and the live page check against a stand-in fetch. No Browser Use and no airline is contacted."""

import httpx

from app.tasks import checkin_desk, checkin_pages


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
