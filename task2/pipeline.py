import json
import sys

import pandas as pd
import random

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

    client = LLMClient(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    result = dict()

    temp = random.randint(0, 100)
    for i in range(temp * 100, (temp + 1) * 100): 
        # делаю для случайной сотни комментов так как для 150_000 долго, 
        # там скорее всего в какой-то момент лимиты на запросы включаются и они дольше проходят, 
        # типа token-buckets что-то наверно. Не погружался
        code, comment = df.iloc[i, 0], df.iloc[i, 1]

        prompt = PROMPT + comment
        result[code] = client.analyze(prompt)

        if (i + 1) % 10 == 0:
            print(f"Итерация № {i - (temp * 100) + 1}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


    print("Результаты сохранены.")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
