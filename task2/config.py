import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "comments.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "results", "analysis_result.json")

ALLOWED_VERDICTS = ("toxic", "non-toxic", "uncertain")

SYSTEM_PROMPT = (
    "Ты — модератор онлайн-комментариев. Оцениваешь уровень токсичности "
    "комментариев пользователей. Отвечай строго в формате JSON по заданной "
    "схеме. Все текстовые поля пиши на русском языке."
)

PROMPT = (
    "Проанализируй текст комментария на токсичность.\n\n"
    "Верни строго JSON по схеме:\n"
    '{"verdict": "toxic" | "non-toxic" | "uncertain", "reason": "<краткое обоснование на русском>"}\n\n'
    "Допустимые значения verdict — ровно эти три строки на английском, без вариаций. "
    "Никаких других ключей в ответе быть не должно. "
    "uncertain — только если по тексту реально нельзя решить.\n\n"
    "Примеры:\n"
    'Комментарий: "You are an idiot"\n'
    'Ответ: {"verdict": "toxic", "reason": "Прямое оскорбление личности"}\n\n'
    'Комментарий: "Thanks for fixing this"\n'
    'Ответ: {"verdict": "non-toxic", "reason": "Нейтральная благодарность"}\n\n'
    'Комментарий: "ok lol"\n'
    'Ответ: {"verdict": "uncertain", "reason": "Слишком мало контекста для оценки"}\n\n'
    "Комментарий:\n"
)
