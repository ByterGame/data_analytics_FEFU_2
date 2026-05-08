import io
import logging
import re
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import agent
from .config import GROQ_API_KEY, MAX_FILE_SIZE_MB
from .injection import sanitize_instruction

_SECTION_HEADER = re.compile(r"^[А-ЯЁA-Z][А-ЯЁA-Z0-9 \-—:]{2,}$")
_BULLET_PREFIXES = ("-", "*", "•")
_SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def parse_agent_report(text: str) -> list[dict]:
    """Парсит текстовый отчёт агента в список секций {title, bullets}."""
    if not text:
        return []
    sections: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if _SECTION_HEADER.match(stripped):
            if current and (current["bullets"] or current["title"]):
                sections.append(current)
            current = {"title": stripped.rstrip(":"), "bullets": []}
            continue
        if current is None:
            current = {"title": "", "bullets": []}
        if stripped.startswith(_BULLET_PREFIXES):
            current["bullets"].append(stripped.lstrip("-*• ").strip())
        elif current["bullets"]:
            current["bullets"][-1] += " " + stripped
        else:
            current["bullets"].append(stripped)
    if current and (current["bullets"] or current["title"]):
        sections.append(current)
    return sections


logger = logging.getLogger("task3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Data Analytics Agent")

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def read_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contents))

    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contents), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("Не удалось определить кодировку CSV-файла.")


def _index_error(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"error": message})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/result")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    instruction: str = Form(""),
):
    if not GROQ_API_KEY:
        return _index_error(request, "API-ключ Groq не настроен. Добавьте GROQ_API_KEY в .env файл.")

    if not file.filename.endswith(_SUPPORTED_EXTENSIONS):
        return _index_error(request, "Поддерживаемые форматы: CSV, XLSX.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return _index_error(request, f"Файл слишком большой ({size_mb:.1f} MB). Максимум: {MAX_FILE_SIZE_MB} MB.")

    try:
        df = read_uploaded_file(contents, file.filename)
    except Exception as e:
        return _index_error(request, f"Ошибка чтения файла: {e}")

    if len(df) == 0:
        return _index_error(request, "Файл пуст.")

    cleaned_instruction, suspicious = sanitize_instruction(instruction)

    report_text: str | None = None
    charts: list[str] = []
    trace: list[dict] = []
    llm_error: str | None = None

    try:
        report_text, charts, trace = agent.run(
            df=df,
            instruction=cleaned_instruction,
            suspicious=suspicious,
        )
    except Exception as e:
        logger.exception("agent run failed")
        llm_error = f"Ошибка агента: {e}"

    tool_calls_count = sum(1 for t in trace if t.get("type") == "tool_call")
    logger.info(
        "agent done: file=%s rows=%d cols=%d tool_calls=%d charts=%d suspicious=%s",
        file.filename, len(df), df.shape[1], tool_calls_count, len(charts), suspicious,
    )

    return templates.TemplateResponse(request, "results.html", {
        "filename": file.filename,
        "rows": len(df),
        "cols": df.shape[1],
        "instruction": cleaned_instruction,
        "suspicious": suspicious,
        "report": report_text,
        "report_sections": parse_agent_report(report_text or ""),
        "charts": charts,
        "trace": trace,
        "tool_calls_count": tool_calls_count,
        "llm_error": llm_error,
        "created_at": datetime.now(),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)
