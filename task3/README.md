# Задание 3. Мини-продукт: AI Data Analytics

## Описание

Веб-приложение на FastAPI для автоматического анализа данных с помощью ИИ. Пользователь загружает CSV или Excel файл и получает:

- Описательную статистику (средние, медианы, квартили, пропуски)
- Интерактивные графики (гистограммы, корреляции, выбросы)
- AI-инсайты от Groq (находки, аномалии, рекомендации)

---

**Ссылка:** [data-analytics-fefu-2](https://data-analytics-fefu-2.onrender.com)

## Технологии

- **Backend:** FastAPI + Jinja2
- **Визуализация:** Plotly (интерактивные графики)
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **Деплой:** Render (Docker)

## Запуск локально

### 1. Установка зависимостей

```bash
cd task3
pip install -r requirements.txt
```

### 2. Настройка API-ключа

```bash
cp .env.example .env
# Отредактируйте .env и укажите API_KEY
```

Получите бесплатный API-ключ на [console.groq.com](https://console.groq.com/keys).

### 3. Запуск

```bash
uvicorn app.main:app --reload
```

Откройте http://localhost:8000 в браузере.

### 4. Использование

1. Перетащите CSV/Excel файл на страницу загрузки
2. Нажмите "Анализировать"
3. Дождитесь результатов (обычно 5-15 секунд)
4. Изучите статистику, графики и AI-инсайты

## Деплой на Render

1. Push код на GitHub
2. Создайте Web Service на [render.com](https://render.com)
3. Root Directory: `task3/`
4. Environment: Docker
5. Добавьте переменную `GROQ_API_KEY`

## Структура проекта

```
task3/
  app/
    main.py            # FastAPI приложение
    config.py          # Настройки
    llm_client.py      # API клиент
    analytics.py       # Pandas-анализ
    chart_generator.py # Plotly-графики
    templates/         # HTML-шаблоны
    static/            # CSS
  Dockerfile
  requirements.txt
```

## Пример

Тестовый датасет для проверки: `data/diabetes_sample.csv` (200 строк, предсказание диабета).
