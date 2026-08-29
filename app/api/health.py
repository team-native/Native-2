"""상태 점검 endpoint."""
from fastapi import APIRouter
from app.schemas.analysis import HealthResponse
from app.services.inference_service import InferenceService
router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="0.2.0", model_ready=InferenceService().is_ready())
