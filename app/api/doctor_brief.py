from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_secret
from app.db.session import get_db
from app.llm.provider import chat_invoke

router = APIRouter(prefix="/doctor-brief", tags=["doctor-brief"], dependencies=[Depends(verify_api_secret)])

SYSTEM_PROMPT = """You are Saheli preparing a concise Doctor / clinical stakeholder brief for a family doctor or care team.

Audience: clinician or allied health professional — factual, telegraphic, no caregiver small-talk.

Rules:
- Use ONLY facts from the timeline / memory provided.
- No diagnosis. No treatment recommendations. No invented labs or vitals.
- Prefer dated bullets. Quote printed values exactly when present.
- Flag gaps explicitly ("not recorded").
- Never alarmist; never casual.

Structure:
1. Identifying context (name only — no PHI beyond what is given)
2. Recent vitals / labs (values + dates only)
3. Medications & adherence signals (reported only; with what each medicine is for, the prescribing doctor and the
   record it came from when the care record gives them)
4. Symptoms / check-ins (quoted or paraphrased factually)
5. Orders / logistics only if clinically relevant (e.g. missed pharmacy pickup)
6. Open questions / missing data for the clinician

End with: Reported only — not a clinical assessment."""


class DoctorBriefRequest(BaseModel):
    subject_name: str = Field(..., min_length=1)
    timeline: str = Field(..., min_length=1)
    stale_health: str = ""
    memory_profile: str = ""
    # Kavach ids: the brief then also reads the person's current care record with its links (D-13)
    family_id: str | None = Field(default=None, max_length=80)
    subject_id: str | None = Field(default=None, max_length=64)


async def care_record_lines(session: AsyncSession, family_id: str, subject_id: str) -> str:
    """The current medicines, conditions and allergies, each medicine with what it is for, who prescribed it and the
    record it came from: 'Metformin 500 mg at 08:00 — for diabetes; prescribed by Dr Sharma; from the 3 Sep prescription'."""
    from app.care import links, store
    from app.care.digest import link_text

    facts = [f for f in await store.facts(session, family_id, subject_id, statuses=("active",))
             if f.domain in ("medicine", "condition", "allergy", "vital_target")]
    linked = links.by_from([ln for ln in await links.links_for(session, family_id, subject_id, statuses=("active",))])
    titles = {"medicine": "Medicine", "condition": "Condition", "allergy": "Allergy", "vital_target": "Target"}
    return "\n".join(f"- {titles[f.domain]}: {f.text}{link_text(linked.get(f.key))}" for f in sorted(facts, key=lambda x: (x.domain, x.key)))


class DoctorBriefResponse(BaseModel):
    brief: str
    audience: str = "doctor"


@router.post("/generate", response_model=DoctorBriefResponse)
async def generate_doctor_brief(body: DoctorBriefRequest, session: Annotated[AsyncSession, Depends(get_db)]):
    extras = []
    if body.family_id and body.subject_id:
        record = await care_record_lines(session, body.family_id, body.subject_id)
        if record:
            extras.append(f"Current care record (reported; medicines with what each is for, who prescribed it, which record):\n{record[:6000]}")
    if body.stale_health.strip():
        extras.append(
            "Health memory flagged for review (list as open questions, not diagnosis):\n"
            f"{body.stale_health[:4000]}"
        )
    if body.memory_profile.strip():
        extras.append(f"Curated memory profile (reported only):\n{body.memory_profile[:6000]}")
    extra_block = ("\n\n" + "\n\n".join(extras)) if extras else ""
    user = f"Patient/care recipient: {body.subject_name}\n\nCare timeline:\n{body.timeline[:12000]}{extra_block}"
    text = await chat_invoke(SYSTEM_PROMPT, user)
    return DoctorBriefResponse(brief=text.strip(), audience="doctor")
