# Задание 2. API-пайплайн: данные → LLM → результат

## Описание

Python-скрипт, который автоматически анализирует токсичность комментариев с помощью LLM.

**Задача:** анализирование уровня токсичности комментария.

### Пайплайн

```  
results/analysis_result.json
    ^
    | Сохранение
    |
Структурированный JSON (анализ уровня токсичности каждого отдельного комментария)
    ^ 
    | Groq API
    |
CSV-файл (комментарии)
```

## Датасет

[toxic-comment-classification-challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge?select=test.csv) — ~150_000 коментариев для оценки, в скрипте оцениваются случайные 100 комментариев.

## Инструкция запуска

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка API-ключа

Получите бесплатный API-ключ на [console.groq.com](https://console.groq.com/keys).

```bash
cp .env.example .env
# Отредактируйте .env и вставьте ваш API_KEY
```

### 3. Запуск

```bash
python pipeline.py
```

### 4. Результат

- `results/analysis_result.json` — структурированный JSON от LLM

## Пример входных данных

Файл `data/comments.csv`:

| id | comment_text  |
|----|---------------|
| 0  | You're so bad |
| 1  | You're so good| 

## Пример выходных данных

```json
{
  "05b4090912f09fe9": {
    "результат": "токсичный"
  },
  "05b48cab97335fb2": {
    "результат": "токсичный"
  },
  "05b4bdd800a00c67": {
    "результат": "не токсичный"
  },
}
```

## Технологии

- Python 3.10+
- Groq API (llama-3.3-70b-versatile)
- Pandas

## Примечание

Подключитесь к VPN перед запуском скрипта
