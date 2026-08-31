"""모델 기반 분석 endpoint."""
from fastapi import APIRouter, HTTPException
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse, SmishingAnalyzeRequest, SmishingAnalyzeResponse, SmishingChatRequest, SmishingChatResponse
from app.llm.service import LLMNotConfiguredError, LLMServiceError, OpenAIExplanationService
from app.services.inference_service import InferenceService, ModelNotReadyError
router = APIRouter(tags=["analysis"])
_service = InferenceService()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_transfer(request: AnalyzeRequest) -> AnalyzeResponse:
    try: return AnalyzeResponse(**_service.analyze(request))
    except ModelNotReadyError as error: raise HTTPException(status_code=503, detail="model_not_ready") from error

@router.post("/analyze/smishing", response_model=SmishingAnalyzeResponse)
def analyze_smishing(request: SmishingAnalyzeRequest) -> SmishingAnalyzeResponse:
    try: return SmishingAnalyzeResponse(**_service.analyze_smishing(request))
    except ModelNotReadyError as error: raise HTTPException(status_code=503, detail="model_not_ready") from error

@router.post("/chat/smishing", response_model=SmishingChatResponse)
def chat_about_smishing(request: SmishingChatRequest) -> SmishingChatResponse:
    try:
        result = _service.analyze_smishing(request)
        answer = OpenAIExplanationService().explain(result)
        return SmishingChatResponse(**result, answer=answer)
    except ModelNotReadyError as error: raise HTTPException(status_code=503, detail="model_not_ready") from error
    except LLMNotConfiguredError as error: raise HTTPException(status_code=503, detail="llm_not_configured") from error
    except LLMServiceError as error: raise HTTPException(status_code=502, detail=f"llm_unavailable: {error}") from error
