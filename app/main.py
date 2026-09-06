from fastapi import FastAPI

from app.api.routes.alerts import router as alerts_router
from app.api.routes.assets import router as assets_router
from app.api.routes.health import router as health_router
from app.api.routes.predictions import router as predictions_router
from app.api.routes.schedules import router as schedules_router
from app.core.config import settings

app = FastAPI(title=settings.project_name)

app.include_router(health_router)
app.include_router(assets_router, prefix=settings.api_v1_prefix)
app.include_router(predictions_router, prefix=settings.api_v1_prefix)
app.include_router(schedules_router, prefix=settings.api_v1_prefix)
app.include_router(alerts_router, prefix=settings.api_v1_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.project_name,
        "environment": settings.environment,
        "docs": "/docs",
    }
