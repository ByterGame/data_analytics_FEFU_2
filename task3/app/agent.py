import json
import logging

import pandas as pd
from groq import BadRequestError

from .config import MAX_AGENT_ITERATIONS
from .llm_client import LLMClient
from .prompts import PYTHON_EXEC_TOOL, SYSTEM_PROMPT, build_user_message
from .sandbox import run_code

logger = logging.getLogger("task3.agent")


class AgentRunError(RuntimeError):
    pass


def _execute_tool_call(tc, df, charts, trace, iteration, step_num) -> str:
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if name != "python_exec":
        trace.append({
            "step": step_num,
            "iteration": iteration,
            "type": "unknown_tool",
            "name": name,
        })
        return f"ОШИБКА: неизвестный инструмент {name!r}."

    code = args.get("code", "")
    output = run_code(code, df, charts)
    trace.append({
        "step": step_num,
        "iteration": iteration,
        "type": "tool_call",
        "code": code,
        "output_preview": output[:300],
    })
    return output


def _force_final(client: LLMClient, messages: list[dict], trace: list[dict]) -> str:
    messages.append({
        "role": "user",
        "content": (
            "Лимит вызовов инструмента исчерпан. На основе уже собранных данных "
            "выдай финальный отчёт по заданному формату — без новых вызовов python_exec."
        ),
    })
    response = client.chat(messages=messages, tools=None)
    trace.append({"iteration": MAX_AGENT_ITERATIONS + 1, "type": "forced_final"})
    return (response.choices[0].message.content or "").strip()


def run(
    df: pd.DataFrame,
    instruction: str,
    suspicious: bool,
    client: LLMClient | None = None,
) -> tuple[str, list[str], list[dict]]:
    """
    Запускает агентский цикл tool-calling.
    Возвращает (report_text, charts_b64, trace).
    `trace` — лог итераций для отладки/верификации (что вызывал агент).
    """
    client = client or LLMClient()
    charts: list[str] = []
    trace: list[dict] = []
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(df, instruction, suspicious)},
    ]

    final_text = ""
    step_num = 0

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        try:
            response = client.chat(
                messages=messages,
                tools=[PYTHON_EXEC_TOOL],
                tool_choice="auto",
            )
        except BadRequestError as e:
            logger.warning("groq tool_use_failed on iter %d: %s", iteration, e)
            trace.append({
                "iteration": iteration,
                "type": "model_error",
                "message": "Модель вернула некорректный tool_call. Переходим к финальному отчёту.",
            })
            break

        msg = response.choices[0].message
        tool_call = msg.tool_calls[0] if msg.tool_calls else None

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if tool_call:
            assistant_entry["tool_calls"] = [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }]
        messages.append(assistant_entry)

        if tool_call is None:
            final_text = (msg.content or "").strip()
            trace.append({"iteration": iteration, "type": "final"})
            break

        step_num += 1
        output = _execute_tool_call(tool_call, df, charts, trace, iteration, step_num)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": output,
        })

    if not final_text:
        final_text = _force_final(client, messages, trace)

    if not final_text:
        raise AgentRunError("Агент не вернул финальный отчёт.")

    return final_text, charts, trace
