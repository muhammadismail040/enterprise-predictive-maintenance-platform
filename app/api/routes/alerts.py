from fastapi import APIRouter

from app.services.mock_data import ALERTS

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts() -> dict[str, list[dict[str, str]]]:
    return {"items": ALERTS}
