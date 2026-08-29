from fastapi import FastAPI
from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
app = FastAPI(title="안전 송금 금융사기 탐지 AI", version="0.2.0")
app.include_router(analyze_router)
app.include_router(health_router)
