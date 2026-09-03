from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_api_secret
from app.llm.provider import chat_invoke

router = APIRouter(prefix="/care-brief", tags=["care-brief"], dependencies=[Depends(verify_api_secret)])

SYSTEM_PROMPT = """You are Saheli, writing a Care Brief for a family caregiver.

Use ONLY facts from the timeline provided. No diagnosis. No invented values.
Companion tone, gender-neutral, calm — never alarmist.

Structure with short sections:
- How they are (last check-in / messages)
- Medicines & schedule (confirmed vs open)
- Vitals (printed values only)
- Orders & approvals
- Quiet context (context_signal items — factual, not urgent)

End with: Reported only — nothing invented."""


class CareBriefRequest(BaseModel):
    subject_name: str = Field(..., min_length=1)
    timeline: str = Field(..., min_length=1)


class CareBriefResponse(BaseModel):
    brief: str


@router.post("/generate", response_model=CareBriefResponse)
async def generate_care_brief(body: CareBriefRequest):
    user = f"Care recipient: {body.subject_name}\n\nTimeline:\n{body.timeline[:12000]}"
    text = await chat_invoke(SYSTEM_PROMPT, user)
    return CareBriefResponse(brief=text.strip())
