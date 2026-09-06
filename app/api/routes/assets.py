from fastapi import APIRouter

from app.services.mock_data import ASSETS

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("")
def list_assets() -> dict[str, list[dict[str, str]]]:
    return {"items": ASSETS}
