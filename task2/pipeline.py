import json
import sys

import pandas as pd

from config import GROQ_API_KEY, INPUT_FILE, MODEL_NAME, OUTPUT_JSON, PROMPT
from llm_client import LLMClient


def main():
    if not GROQ_API_KEY:
        print("Ошибка: установите GROQ_API_KEY в файле .env")
        print("Скопируйте .env.example в .env и укажите ваш API-ключ.")
        print("Ключ можно получить бесплатно на https://console.groq.com/keys")
        sys.exit(1)

    print(f"Чтение данных из {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов.")

    stats = df.describe(include="all").to_string()
    prompt = PROMPT + "Статистика датасета:\n" + stats
    print(f"Отправка запроса в Groq ({MODEL_NAME})...")
    client = LLMClient(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    result = client.analyze(prompt)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


    print("Результаты сохранены.")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
