"""향후 LLM 설명 프롬프트 정의 위치."""


def build_explanation_prompt(risk_result: dict[str, object]) -> str:
    """TODO: Risk Engine 결과만 쉬운 언어로 설명하도록 프롬프트를 구성한다.

    LLM에 원문이나 금융사기 판정 권한을 주지 않는다. 현재는 외부
    호출을 하지 않으므로 이 함수는 아직 구현하지 않는다.
    """
    raise NotImplementedError("LLM prompt generation is not implemented in v0.1")
