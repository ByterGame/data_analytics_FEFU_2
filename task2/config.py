import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "comments.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "results", "analysis_result.json")


PROMPT = (
        "Проанализируй следующий комментарий на токсичность "
        "В ответе выбирай из трех вариантов: 1) точно токсичный. 2) точно не токсичный."
        "3) Описание настроения, если не уверен в токсичности\n"
        "Текст комментария:\n"
        # текст добавляется непосредственно перед отправкой к llm
    )