import base64
import io
import re
import sys
import threading
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EXEC_TIMEOUT_SEC, MAX_CHARTS, MAX_TOOL_OUTPUT_CHARS

_BLOCKED_PATTERNS = [re.compile(p) for p in [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+pathlib\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+multiprocessing\b",
    r"\bimport\s+matplotlib\b",
    r"\bimport\s+seaborn\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+sys\b",
    r"\bfrom\s+subprocess\b",
    r"\bfrom\s+socket\b",
    r"\bfrom\s+shutil\b",
    r"\bfrom\s+pathlib\b",
    r"\bfrom\s+multiprocessing\b",
    r"\bfrom\s+matplotlib\b",
    r"\bfrom\s+seaborn\b",
    r"\b__import__\b",
    r"\b__class__\b",
    r"\b__subclasses__\b",
    r"\b__bases__\b",
    r"\b__globals__\b",
    r"\b__builtins__\b",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"\.read_csv\s*\(",
    r"\.read_excel\s*\(",
    r"\.read_json\s*\(",
    r"\.to_csv\s*\(",
    r"\.to_excel\s*\(",
    r"\.to_json\s*\(",
    r"\.to_pickle\s*\(",
    r"\.read_pickle\s*\(",
]]

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "frozenset": frozenset, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "len": len, "list": list, "map": map, "max": max, "min": min, "next": next,
    "object": object, "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set, "slice": slice, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip, "True": True,
    "False": False, "None": None, "hasattr": hasattr, "id": id, "hex": hex, "oct": oct,
    "bin": bin, "chr": chr, "ord": ord, "iter": iter, "format": format,
    "complex": complex, "bytes": bytes, "bytearray": bytearray, "callable": callable,
    "ZeroDivisionError": ZeroDivisionError, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError,
    "Exception": Exception, "ArithmeticError": ArithmeticError,
    "AttributeError": AttributeError, "NameError": NameError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
}


def is_code_safe(code: str) -> tuple[bool, str]:
    """Проверяет код регексами на запрещённые конструкции."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return False, f"запрещённый паттерн: {pattern.pattern}"
    return True, ""


def _capture_charts(charts: list[str]) -> int:
    """Сохраняет все открытые matplotlib figures в charts как base64 PNG."""
    added = 0
    fig_nums = list(plt.get_fignums())
    for num in fig_nums:
        if len(charts) >= MAX_CHARTS:
            break
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
            buf.seek(0)
            charts.append(base64.b64encode(buf.read()).decode("ascii"))
            added += 1
        except Exception:
            pass
    plt.close("all")
    return added


def run_code(code: str, df: pd.DataFrame, charts: list[str]) -> str:
    """
    Выполняет код LLM в sandbox с таймаутом.
    Графики (matplotlib figures) дописываются в charts как base64-PNG.
    Возвращает текст для tool-сообщения LLM.
    """
    safe, reason = is_code_safe(code)
    if not safe:
        return (
            f"ОТКАЗАНО: код содержит {reason}.\n"
            "ВАЖНО: в неймспейсе УЖЕ доступны `df`, `pd`, `np`, `plt` — НЕ ПИШИ `import` совсем. "
            "Просто строй графики через `plt.figure(); plt.hist(df['col'], ...)` без импортов. "
            "Перепиши код без любых import-ов и попробуй снова."
        )

    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "__name__": "_sandbox_",
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
    }

    result_holder = {"stdout": "", "error": None}
    stdout_buf = io.StringIO()

    def _target():
        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        try:
            exec(code, namespace)
        except Exception:
            result_holder["error"] = traceback.format_exc(limit=3)
        finally:
            sys.stdout = old_stdout
            result_holder["stdout"] = stdout_buf.getvalue()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT_SEC)

    if thread.is_alive():
        plt.close("all")
        return (
            f"ТАЙМАУТ: код выполнялся дольше {EXEC_TIMEOUT_SEC} с и был остановлен. "
            "Упрости вычисление или уменьши объём данных."
        )

    added_charts = _capture_charts(charts)

    parts = []
    stdout_text = result_holder["stdout"].strip()
    if stdout_text:
        parts.append(f"STDOUT:\n{stdout_text}")
    if result_holder["error"]:
        parts.append(f"EXCEPTION:\n{result_holder['error'].strip()}")
    if added_charts:
        parts.append(f"[сохранено графиков: {added_charts}; всего: {len(charts)}/{MAX_CHARTS}]")
    if not parts:
        parts.append("(нет вывода — используй print() для печати результатов)")

    output = "\n\n".join(parts)
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [обрезано до {MAX_TOOL_OUTPUT_CHARS} символов]"
    return output
