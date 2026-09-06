from fastapi import APIRouter

from app.services.mock_data import PREDICTIONS

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("")
def list_predictions() -> dict[str, list[dict[str, str | float | int]]]:
    return {"items": PREDICTIONS}
