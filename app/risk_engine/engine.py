"""향후 Risk Engine의 계약만 정의한다. 현재 점수화하지 않는다."""
from __future__ import annotations

from typing import Any, Protocol


class RiskEngine(Protocol):
    """fraud_probability와 정책 근거를 향후 위험도로 계산할 인터페이스.

    TODO: model explanation, fraud_probability 및 안전 정책을 입력받아
    ``risk_score``(0~100), ``risk_level``, ``reasons``를 반환한다. 현재는
    fraud_probability * 100 같은 임의 점수화를 하지 않는다.
    """

    def evaluate(
        self,
        fraud_prediction: dict[str, Any],
        model_explanation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """TODO: 향후 위험 점수와 근거를 계산한다."""
