import json
import random
import time

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

from config import ALLOWED_VERDICTS, SYSTEM_PROMPT


_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_MAX_RETRIES = 6
_BASE_DELAY = 1.0


def _validate(data) -> dict | None:
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if not isinstance(verdict, str):
        return None
    verdict = verdict.strip().lower()
    if verdict not in ALLOWED_VERDICTS:
        return None
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return {"verdict": verdict, "reason": reason}


class LLMClient:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _request(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)

    def analyze(self, prompt: str) -> dict:
        last_error: str | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                data = self._request(prompt)
            except _RETRYABLE as e:
                last_error = f"{type(e).__name__}: {e}"
            except json.JSONDecodeError as e:
                last_error = f"JSONDecodeError: {e}"
            else:
                validated = _validate(data)
                if validated is not None:
                    return validated
                last_error = f"схема не соответствует: {data}"

            if attempt < _MAX_RETRIES - 1:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        return {
            "verdict": "uncertain",
            "reason": f"не удалось получить корректный ответ после {_MAX_RETRIES} попыток: {last_error}",
        }
