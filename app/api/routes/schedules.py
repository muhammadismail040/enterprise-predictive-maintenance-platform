from fastapi import APIRouter

from app.services.mock_data import SCHEDULES

router = APIRouter(prefix="/schedules", tags=["scheduling"])


@router.get("")
def list_schedules() -> dict[str, list[dict[str, str]]]:
    return {"items": SCHEDULES}
