"""Admin "watch live" (founder 2026-10-13): conversations of the last half hour with pace, cost and mood, running
orders, check-ins due and today's alerts. No message text in the feed; sims and shadow families are left out."""

from datetime import timedelta

from app.api import admin
from app.care import mood, store
from app.core import clock


async def test_the_live_feed_shows_who_is_talking_and_waiting_without_their_words(db, at):
    at("2026-10-13 10:00")
    me = {"id": "self-l", "name": "Arjun", "role": "primary caregiver"}
    await store.save_roster(db, "fam-live", me, [me])
    await store.add_turn(db, family_id="fam-live", thread_id=me["id"], role="user", text="secret words about my day")
    await store.add_turn(db, family_id="fam-live", thread_id=me["id"], role="assistant", text="reply")
    await store.add_turn(db, family_id="fam-live", thread_id=me["id"], role="user", text="and one more")
    await store.add_turn(db, family_id="sim-x", thread_id="p", role="user", text="sim")
    await mood.record(db, family_id="fam-live", subject_id=me["id"], score=2, word="tired", summary="long day")
    await store.open_loop(db, family_id="fam-live", subject_id=me["id"], owner_id=me["id"], kind="activity", title="Gym check-in",
                          wake_at=clock.now() + timedelta(minutes=20))
    at("2026-10-13 10:02")
    out = await admin.admin_live(db)
    assert [c["familyId"] for c in out["conversations"]] == ["fam-live"], "sims are left out"
    c = out["conversations"][0]
    assert c["messages"] == 2 and c["replies"] == 1 and c["waitingForSaheli"] and c["active"]
    assert c["mood"] == {"score": 2, "word": "tired"}
    assert "secret" not in str(out), "no message text in the live feed"
    assert out["comingUp"][0]["kind"] == "activity"
