import io
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analytics import compute_statistics
from .chart_generator import generate_charts
from .config import GROQ_API_KEY, MAX_FILE_SIZE_MB, MAX_ROWS_FOR_CHARTS
from .llm_client import LLMClient

app = FastAPI(title="AI Data Analytics")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")



def read_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    """Читает загруженный файл в DataFrame."""
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(io.BytesIO(contents))

    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contents), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("Не удалось прочитать файл. Поддерживаемые форматы: CSV, XLSX.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/result")
async def upload_file(request: Request, file: UploadFile = File(...)):

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Файл слишком большой ({size_mb:.1f} MB). Максимум: {MAX_FILE_SIZE_MB} MB.",
        })

    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Поддерживаемые форматы: CSV, XLSX.",
        })

    try:
        df = read_uploaded_file(contents, file.filename)
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Ошибка чтения файла: {e}",
        })

    if len(df) == 0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Файл пуст.",
        })

    # Вычисление статистики
    stats = compute_statistics(df)
    sampled = len(df) > MAX_ROWS_FOR_CHARTS
    sample_note = f"Для графиков использована выборка {MAX_ROWS_FOR_CHARTS:,} из {len(df):,} строк." if sampled else None

    # Генерация графиков
    charts = generate_charts(df, max_rows=MAX_ROWS_FOR_CHARTS)

    insights = None
    llm_error = None
    if GROQ_API_KEY:
        try:
            client = LLMClient(api_key=GROQ_API_KEY)
            insights = client.analyze(stats)
        except Exception as e:
            llm_error = f"Ошибка LLM: {e}"
    else:
        llm_error = "API-ключ Groq не настроен. Добавьте GROQ_API_KEY в .env файл."

    result = {
        "filename": file.filename,
        "stats": stats,
        "charts": charts,
        "insights": insights,
        "llm_error": llm_error,
        "sample_note": sample_note,
        "created_at": datetime.now(),
    }

    return templates.TemplateResponse("results.html", {
        "request": request,
        **result,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
