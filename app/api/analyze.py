"""모델 기반 분석 endpoint."""
from fastapi import APIRouter, HTTPException
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.inference_service import InferenceService, ModelNotReadyError
router = APIRouter(tags=["analysis"])
_service = InferenceService()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_transfer(request: AnalyzeRequest) -> AnalyzeResponse:
    try: return AnalyzeResponse(**_service.analyze(request))
    except ModelNotReadyError as error: raise HTTPException(status_code=503, detail="model_not_ready") from error
