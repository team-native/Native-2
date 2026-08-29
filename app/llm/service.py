"""향후 설명용 LLM service 계약."""

from typing import Protocol


class ExplanationService(Protocol):
    """Risk Engine 결과를 고연령층 친화적 설명으로 바꿀 향후 인터페이스.

    TODO: risk_score, risk_level, reasons만 입력으로 사용한다. LLM은 사기 여부나
    위험 점수를 판단하지 않으며 v0.1에서 외부 API를 호출하지 않는다.
    """

    def explain(self, risk_result: dict[str, object]) -> str:
        """TODO: 위험 수준, 이유, 권장 대응을 설명한다."""
