import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
INPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "diabetes_sample.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "results", "analysis_result.json")


PROMPT = (
        "Проанализируй следующий датасет по предсказанию диабета "
        "и верни JSON со следующей структурой:\n"
        "{\n"
        '  "dataset_summary": "краткое описание датасета (2-3 предложения)",\n'
        '  "risk_factors": [\n'
        '    {"factor": "название признака", "risk_level": "high/medium/low", '
        '"evidence": "обоснование на основе данных", "recommendation": "рекомендация"}\n'
        "  ],\n"
        '  "key_findings": ["находка 1", "находка 2", ...],\n'
        '  "demographic_insights": ["инсайт 1", "инсайт 2", ...],\n'
        '  "modeling_recommendations": "рекомендации по построению модели"\n'
        "}\n\n"
    )