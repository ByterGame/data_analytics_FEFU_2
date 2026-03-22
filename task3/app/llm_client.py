import json
from groq import Groq


class LLMClient:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def analyze(self, stats: dict) -> dict:
        """Отправляет статистику в Groq и возвращает JSON-анализ."""
        prompt = (
            "Проанализируй следующую статистику по загруженному датасету "
            "и верни JSON со следующей структурой:\n"
            "{\n"
            '  "summary": "краткое описание датасета и его содержимого (3-4 предложения)",\n'
            '  "key_metrics": [\n'
            '    {"metric": "название метрики", "value": "значение", "interpretation": "интерпретация"}\n'
            "  ],\n"
            '  "findings": ["находка 1", "находка 2", ...],\n'
            '  "anomalies": ["аномалия 1", ...],\n'
            '  "recommendations": ["рекомендация 1", ...]\n'
            "}\n\n"
            f"Статистика:\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — аналитик данных. Анализируй предоставленную статистику по датасету. "
                        "Отвечай строго в формате JSON. Все текстовые поля пиши на русском языке."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
