from groq import Groq

from .config import GROQ_API_KEY, MODEL_NAME


class LLMClient:
    """Тонкая обёртка над groq.Groq().chat.completions.create() с поддержкой tools."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.client = Groq(api_key=api_key or GROQ_API_KEY)
        self.model = model or MODEL_NAME

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = False
        return self.client.chat.completions.create(**kwargs)
