"""Saheli sings: "gaake sunao" sends a short song as a voice note (send_song), never a switch of the voice setting
(live 2026-10-09: Maa asked for a bhajan sung, Saheli said she cannot sing and turned voice replies to always)."""

import json

from app.brain import persona, tools
from app.brain.host import WRITE_TOOLS, ShadowHost
from app.care import store
from app.sim.world import SimHost

FAM = "fam-sing"
ELDER = {"id": "e-sing", "name": "Vasundara", "role": "elder"}
SON = {"id": "c-sing", "name": "Kritarth", "role": "primary caregiver"}
BHAJAN = "पायो जी म्हें तो राम रतन धन पायो।\nवस्तु अमोलक दी म्हारे सतगुरु, किरपा कर अपणायो॥"


def c(db, speaker, host):
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker=speaker, members=[ELDER, SON])


async def test_sings_to_the_person_who_asked(db, at):
    at("2026-10-09 18:52")
    host = SimHost()
    out, err = await tools.run(c(db, ELDER, host), "sing", {"lyrics": BHAJAN, "style": "slow Meera bhajan", "title": "पायो जी म्हें तो"})
    assert not err, out
    song = host.world.sent[-1]
    assert song["song"] and song["to"] == ELDER["id"] and song["lyrics"] == BHAJAN and song["style"] == "slow Meera bhajan"
    assert "voice note" in json.loads(out)["note"]
    turns = await store.recent_turns(db, FAM, ELDER["id"], limit=5)
    assert turns[-1].role == "assistant" and BHAJAN in turns[-1].text  # she remembers what she sang
    assert ELDER["id"] not in host.world.voice_modes  # the voice setting is untouched


async def test_caregiver_can_ask_her_to_sing_to_maa(db, at):
    at("2026-10-09 20:00")
    host = SimHost()
    out, err = await tools.run(c(db, SON, host), "sing", {"lyrics": "चंदा मामा दूर के\nपुए पकाएँ बूर के", "style": "soft lullaby", "to": ELDER["id"]})
    assert not err and host.world.sent[-1]["to"] == ELDER["id"]


async def test_short_songs_only_and_only_to_the_family(db, at):
    at("2026-10-09 20:00")
    host = SimHost()
    _, err = await tools.run(c(db, ELDER, host), "sing", {"lyrics": "\n".join(f"line {i}" for i in range(12))})
    assert err
    _, err = await tools.run(c(db, ELDER, host), "sing", {"lyrics": BHAJAN, "to": "stranger"})
    assert err
    _, err = await tools.run(c(db, ELDER, host), "sing", {"lyrics": "  \n "})
    assert err
    system = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=ELDER, speaker={"id": "saheli", "role": "system"}, members=[ELDER, SON])
    _, err = await tools.run(system, "sing", {"lyrics": BHAJAN})
    assert err  # never sings out of the blue on a wake-up
    assert not host.world.sent


async def test_failed_send_tells_the_brain_to_offer_the_words(db, at):
    at("2026-10-09 20:00")

    class Down(SimHost):
        async def call(self, tool, args, **kw):
            return {"delivered": False, "reason": "outside the 24-hour WhatsApp window"} if tool == "send_song" else await super().call(tool, args, **kw)

    out, err = await tools.run(c(db, SON, Down()), "sing", {"lyrics": BHAJAN, "to": ELDER["id"]})
    assert not err and "could not be sent" in json.loads(out)["note"]


async def test_not_sent_in_shadow_mode(db, at):
    at("2026-10-09 20:00")
    assert "send_song" in WRITE_TOOLS
    host = ShadowHost()
    out, err = await tools.run(c(db, ELDER, host), "sing", {"lyrics": BHAJAN})
    assert not err and host.would_have[-1]["tool"] == "send_song"


def test_persona_sings_only_when_asked_and_voice_setting_is_not_for_songs():
    # founder 2026-10-10: sing only if asked to sing; "bhajan batao" (a suggestion) is answered in text
    assert "only when someone asks you to sing" in persona.PERSONA and "never say you cannot sing" in persona.PERSONA
    assert "bhajan batao" in persona.PERSONA and "Never sing on your own" in persona.PERSONA
    assert "never for 'bhajan batao'" in next(s for s in tools.specs() if s.name == "sing").description
    spec = next(s for s in tools.specs() if s.name == "voice_replies")
    assert "sing" in spec.description
