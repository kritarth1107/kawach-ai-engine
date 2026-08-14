import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_api_secret
from app.db.session import get_db
from app.models.entities import MemberRole
from app.services.family import (
    add_family_member,
    create_elder,
    create_family_with_owner,
    list_families_for_user,
)

router = APIRouter(prefix="/families", tags=["families"], dependencies=[Depends(verify_api_secret)])


class CreateFamilyRequest(BaseModel):
    name: str
    owner_external_id: str = Field(description="Clerk user id or external auth id")
    owner_email: str | None = None
    owner_name: str | None = None


class AddMemberRequest(BaseModel):
    external_id: str
    email: str | None = None
    name: str | None = None
    role: MemberRole = MemberRole.family


class CreateElderRequest(BaseModel):
    display_name: str
    slug: str = Field(description="Unique per family, e.g. mother, father")
    preferred_language: str = "hinglish"


@router.post("")
async def create_family(body: CreateFamilyRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    family, user = await create_family_with_owner(
        db,
        family_name=body.name,
        owner_external_id=body.owner_external_id,
        owner_email=body.owner_email,
        owner_name=body.owner_name,
    )
    return {
        "family": {"id": str(family.id), "name": family.name},
        "owner": {"id": str(user.id), "external_id": user.external_id},
    }


@router.get("")
async def get_families(external_id: str, db: Annotated[AsyncSession, Depends(get_db)]):
    families = await list_families_for_user(db, external_id)
    return {"families": [{"id": str(f.id), "name": f.name} for f in families]}


@router.post("/{family_id}/members")
async def add_member(
    family_id: uuid.UUID,
    body: AddMemberRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    member = await add_family_member(
        db,
        family_id=family_id,
        external_id=body.external_id,
        email=body.email,
        name=body.name,
        role=body.role,
    )
    return {"member_id": str(member.id), "role": member.role.value}


@router.post("/{family_id}/elders")
async def add_elder(
    family_id: uuid.UUID,
    body: CreateElderRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        elder = await create_elder(
            db,
            family_id=family_id,
            display_name=body.display_name,
            slug=body.slug,
            preferred_language=body.preferred_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "elder": {
            "id": str(elder.id),
            "display_name": elder.display_name,
            "slug": elder.slug,
        }
    }
