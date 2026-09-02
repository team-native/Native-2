"""OpenAI Responses API를 이용한 공주 설명 챗봇."""
from __future__ import annotations

import os

from typing import Protocol

from openai import OpenAI, OpenAIError

from app.llm.prompt import build_explanation_prompt


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMServiceError(RuntimeError):
    pass


class ExplanationService(Protocol):
    """Risk Engine 결과를 고연령층 친화적 설명으로 바꿀 향후 인터페이스.

    TODO: risk_score, risk_level, reasons만 입력으로 사용한다. LLM은 사기 여부나
    위험 점수를 판단하지 않으며 v0.1에서 외부 API를 호출하지 않는다.
    """

    def explain(self, risk_result: dict[str, object]) -> str:
        """판정 결과를 사용자 친화적으로 설명한다."""


class OpenAIExplanationService:
    """Reads OPENAI_API_KEY only from the server environment."""
    def __init__(self, client: OpenAI | None = None, model: str = "gpt-5-mini") -> None:
        if client is None and not os.environ.get("OPENAI_API_KEY"):
            raise LLMNotConfiguredError("OPENAI_API_KEY is not configured")
        self.client = client or OpenAI()
        self.model = model

    def explain(self, risk_result: dict[str, object]) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=build_explanation_prompt(risk_result),
                input="모델 판정 결과를 설명하고 사용자가 지금 할 안전한 행동을 알려주세요.",
                store=False,
                reasoning={"effort": "minimal"},
                max_output_tokens=500,
            )
        except OpenAIError as error:
            raise LLMServiceError(str(error)) from error
        return response.output_text.strip()
