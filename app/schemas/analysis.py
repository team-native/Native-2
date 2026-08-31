"""공개 API schema."""
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=0, max_length=10000)
    amount: float = Field(ge=0)
    average_amount: float = Field(ge=0)
    recipient_is_new: bool
    recipient_transfer_count: int = Field(ge=0)
    transfers_last_hour: float = Field(ge=0)
    average_transfers_per_hour: float = Field(ge=0)
    transfer_hour: int = Field(ge=0, le=23)

class AnalyzeResponse(BaseModel):
    text_fraud_probability: float = Field(ge=0, le=1)
    fraud_probability: float = Field(ge=0, le=1)
    model_version: str
    risk_engine_applied: bool = False
    llm_applied: bool = False

class SmishingAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    send_hour: int = Field(ge=0, le=23)
    category: str = Field(default="unknown", min_length=1, max_length=100)

class SmishingAnalyzeResponse(BaseModel):
    smishing_probability: float = Field(ge=0, le=1)
    model_version: str
    model_name: str

class SmishingChatRequest(SmishingAnalyzeRequest):
    pass

class SmishingChatResponse(SmishingAnalyzeResponse):
    answer: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model_ready: bool
