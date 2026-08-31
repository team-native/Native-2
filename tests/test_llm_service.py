from types import SimpleNamespace

from app.llm.service import OpenAIExplanationService


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="의심되는 문자입니다. 링크를 누르지 마세요.")


def test_gongju_explains_model_result_without_sending_message_text() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    result = OpenAIExplanationService(client=client).explain({"smishing_probability": 0.91, "model_name": "공주"})
    assert result.startswith("의심되는")
    assert responses.request is not None
    assert responses.request["model"] == "gpt-5-mini"
    assert "원문" not in str(responses.request)
