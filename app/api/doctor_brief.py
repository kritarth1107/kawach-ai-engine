from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_api_secret
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
3. Medications & adherence signals (reported only)
4. Symptoms / check-ins (quoted or paraphrased factually)
5. Orders / logistics only if clinically relevant (e.g. missed pharmacy pickup)
6. Open questions / missing data for the clinician

End with: Reported only — not a clinical assessment."""


class DoctorBriefRequest(BaseModel):
    subject_name: str = Field(..., min_length=1)
    timeline: str = Field(..., min_length=1)
    stale_health: str = ""
    memory_profile: str = ""


class DoctorBriefResponse(BaseModel):
    brief: str
    audience: str = "doctor"


@router.post("/generate", response_model=DoctorBriefResponse)
async def generate_doctor_brief(body: DoctorBriefRequest):
    extras = []
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
