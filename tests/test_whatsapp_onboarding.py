"""Setup on WhatsApp (founder 2026-10-12: "if a new user sends hi on WhatsApp, start their onboarding directly"): the
setup chat is in the turn while setup is not finished, and the tools add the person cared for, save the speaker's own
name and finish the setup. The account itself is made by the backend (kavach-backend scripts/test-whatsapp-signup.ts)."""

import json

from app.brain import loop, tools
from app.brain.host import WRITE_TOOLS

FAM = "fam-wa-onb"
RAVI = {"id": "ravi-onb", "name": "Friend", "role": "primary caregiver"}


class Host:
    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    async def call(self, tool, args, **kw):
        self.calls.append((tool, args, kw))
        return self.answers.get(tool, {"ok": True})


def self_care_ctx(db, host, speaker=RAVI):
    # a new sign-up's family has nobody else yet: the care record is about the caregiver (as the backend sends it)
    return tools.TurnCtx(session=db, host=host, family_id=FAM, elder=speaker, speaker=speaker, members=[speaker])


async def test_the_setup_chat_is_in_the_turn_until_setup_is_finished(db, at):
    at("2026-10-12 18:00")
    new = loop.TurnRequest(family_id=FAM, elder=RAVI, speaker=RAVI, members=[RAVI], text="hi", onboarding=True)
    dynamic, _ = await loop.turn_context(db, new)
    block = dynamic.split("SETUP NOT FINISHED", 1)[1]
    assert "which language" in block and "add_person" in block and "finish_setup" in block and "my_name" in block
    assert "for themselves (about = their own id)" in block
    done = loop.TurnRequest(family_id=FAM, elder=RAVI, speaker=RAVI, members=[RAVI], text="hi")
    assert "SETUP NOT FINISHED" not in (await loop.turn_context(db, done))[0]
    maa = {"id": "maa-onb", "name": "Kamla", "role": "elder"}
    later = loop.TurnRequest(family_id=FAM, elder=maa, speaker=RAVI, members=[maa, RAVI], text="ok", onboarding=True)
    assert "already added: Kamla (id maa-onb)" in (await loop.turn_context(db, later))[0]


async def test_adding_the_person_they_look_after(db, at):
    at("2026-10-12 18:05")
    host = Host({"add_family_member": {"ok": True, "userId": "maa-onb", "name": "Kamla Devi", "joinLink": "https://wa.me/91900?text=Hi%20Saheli"}})
    c = self_care_ctx(db, host)
    out, err = await tools.run(c, "add_person", {"name": "Kamla Devi", "relationship": "mother", "phone": "+919000000303"})
    got = json.loads(out)
    assert not err and got["id"] == "maa-onb" and got["join_link"].startswith("https://wa.me/")
    tool, args, kw = host.calls[-1]
    assert tool == "add_family_member" and args == {"name": "Kamla Devi", "relationship": "mother", "phone": "+919000000303"}
    assert kw["actor_id"] == RAVI["id"]
    assert c.subject("maa-onb") == "maa-onb", "the rest of this turn can save things about her"
    # once someone is looked after, more people are added on the dashboard; never by the care recipient
    maa = {"id": "maa-onb", "name": "Kamla", "role": "elder"}
    family = tools.TurnCtx(session=db, host=host, family_id=FAM, elder=maa, speaker=RAVI, members=[maa, RAVI])
    _, err = await tools.run(family, "add_person", {"name": "Someone"})
    assert err
    _, err = await tools.run(tools.TurnCtx(session=db, host=host, family_id=FAM, elder=maa, speaker=maa, members=[maa, RAVI]),
                             "add_person", {"name": "Someone"})
    assert err
    refused = Host({"add_family_member": {"ok": False, "error": "that number looks incomplete"}})
    out, err = await tools.run(self_care_ctx(db, refused), "add_person", {"name": "Kamla Devi", "phone": "123"})
    assert err and "incomplete" in json.loads(out)["refused"]


async def test_their_name_and_finishing_setup(db, at):
    at("2026-10-12 18:10")
    host = Host({"finish_setup": {"ok": True, "dashboard": "https://app.kavach.care", "signIn": "with this WhatsApp number"}})
    c = self_care_ctx(db, host)
    out, err = await tools.run(c, "my_name", {"first_name": "Ravi", "last_name": "Kumar"})
    assert not err and host.calls[-1][:2] == ("set_my_name", {"firstName": "Ravi", "lastName": "Kumar"})
    out, err = await tools.run(c, "finish_setup", {})
    assert not err and json.loads(out) == {"result": "done", "dashboard": "https://app.kavach.care", "signIn": "with this WhatsApp number"}
    system = {"id": "saheli", "role": "system"}
    _, err = await tools.run(tools.TurnCtx(session=db, host=host, family_id=FAM, elder=RAVI, speaker=system, members=[RAVI]), "finish_setup", {})
    assert err


def test_setup_writes_never_run_in_shadow_and_the_api_takes_the_flag():
    from app.api.brain import TurnIn

    assert {"add_family_member", "set_my_name", "finish_setup"} <= WRITE_TOOLS
    body = TurnIn.model_validate({"family_id": FAM, "elder": RAVI, "speaker": RAVI, "text": "hi", "onboarding": True})
    assert body.onboarding is True
    assert TurnIn.model_validate({"family_id": FAM, "elder": RAVI, "speaker": RAVI, "text": "hi"}).onboarding is False
