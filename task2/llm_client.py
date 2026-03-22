import json
from groq import Groq


class LLMClient:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def analyze(self, prompt: str) -> dict:
        """Отправляет промпт в Groq и возвращает JSON-ответ."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — аналитик медицинских данных. Анализируй предоставленную статистику "
                        "по датасету предсказания диабета. Отвечай строго в формате JSON. "
                        "Все текстовые поля пиши на русском языке."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
