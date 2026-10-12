"""A Google Maps link anywhere is read for its exact point and place details, and can be saved as a place (founder
2026-10-13: "if someone sends the location as a google map link Saheli should be able to fetch it from there and add, at
least the coordinates and other info")."""

import json

from app.brain import tools
from app.care import store
from app.sim.world import SimHost

FAM = "fam-maps"
ME = {"id": "self-m", "name": "Kittu", "role": "primary caregiver"}
PAGE = ('<html><head><meta content="Minister Rajesh Agrawal Ji M4 · Sector 24, Nava Raipur, Chhattisgarh 492018" '
        'property="og:title"></head></html>')


class Places(SimHost):
    def __init__(self):
        super().__init__()
        self.saved = []

    async def call(self, tool, args, **kw):
        if tool == "save_place":
            self.saved.append(args)
            return {"ok": True, "place": args.get("name")}
        return await super().call(tool, args, **kw)


def ctx(db, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ME, speaker=ME, members=[ME], user_text="https://maps.app.goo.gl/QHByb")


async def test_a_short_link_reads_to_its_point_name_and_address(db, at, monkeypatch):
    async def follow(link):
        return ("https://www.google.com/maps/place/Minister+Rajesh+Agrawal+Ji+M4/@21.1612,81.7871,17z/data=!3d21.16123!4d81.78714", PAGE)

    monkeypatch.setattr(tools, "_follow", follow)
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    out, err = await tools.run(ctx(db, Places()), "map_link", {"link": "https://maps.app.goo.gl/QHByb"})
    got = json.loads(out)
    assert not err and got["read"] and (got["lat"], got["lng"]) == (21.16123, 81.78714)
    assert got["name"] == "Minister Rajesh Agrawal Ji M4" and got["address"] == "Sector 24, Nava Raipur, Chhattisgarh 492018"


async def test_save_place_with_a_link_saves_the_exact_point_and_the_address_the_link_carries(db, at, monkeypatch):
    async def follow(link):
        return ("https://www.google.com/maps/place/M4/@21.16,81.78,17z/data=!3d21.16123!4d81.78714", PAGE)

    monkeypatch.setattr(tools, "_follow", follow)
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    host = Places()
    out, err = await tools.run(ctx(db, host), "save_place", {"name": "Minister's office", "maps_link": "https://maps.app.goo.gl/QHByb"})
    assert not err
    saved = host.saved[0]
    assert (saved["lat"], saved["lng"]) == (21.16123, 81.78714) and saved["name"] == "Minister's office"
    assert "Sector 24, Nava Raipur" in saved["address"]


async def test_an_unreadable_link_is_asked_again(db, at, monkeypatch):
    async def follow(link):
        return ("https://www.google.com/sorry/index", "<html>unusual traffic</html>")

    monkeypatch.setattr(tools, "_follow", follow)
    at("2026-10-13 10:00")
    await store.save_roster(db, FAM, ME, [ME])
    out, err = await tools.run(ctx(db, Places()), "save_place", {"maps_link": "https://maps.app.goo.gl/bad"})
    assert err and "ask_location" in out
    got = json.loads((await tools.run(ctx(db, Places()), "map_link", {"link": "https://example.com/x"}))[0])
    assert got["read"] is False
