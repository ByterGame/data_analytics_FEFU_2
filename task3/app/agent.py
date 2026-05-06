import json
import logging

import pandas as pd
from groq import BadRequestError

from .config import MAX_AGENT_ITERATIONS
from .llm_client import LLMClient
from .prompts import PYTHON_EXEC_TOOL, build_system_prompt, build_user_message
from .sandbox import run_code

logger = logging.getLogger("task3.agent")


class AgentRunError(RuntimeError):
    pass


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
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_message(df, instruction, suspicious)},
    ]

    final_text = ""
    step_num = 0
    aborted = False

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
                "step": step_num + 1,
                "iteration": iteration,
                "type": "model_error",
                "message": "Модель вернула некорректный tool_call. Переходим к финальному отчёту.",
            })
            break
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_entry: dict = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if not tool_calls:
            final_text = (msg.content or "").strip()
            trace.append({"iteration": iteration, "type": "final"})
            break

        for tc in tool_calls:
            step_num += 1
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            if name == "python_exec":
                code = args.get("code", "")
                tool_output = run_code(code, df, charts)
                trace.append({
                    "step": step_num,
                    "iteration": iteration,
                    "type": "tool_call",
                    "code": code,
                    "output_preview": tool_output[:300],
                })
            else:
                tool_output = f"ОШИБКА: неизвестный инструмент {name!r}."
                trace.append({
                    "step": step_num,
                    "iteration": iteration,
                    "type": "unknown_tool",
                    "name": name,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_output,
            })

    if not final_text:
        messages.append({
            "role": "user",
            "content": (
                "Лимит вызовов инструмента исчерпан. На основе уже собранных данных "
                "выдай финальный отчёт по заданному формату — без новых вызовов python_exec."
            ),
        })
        response = client.chat(messages=messages, tools=None)
        final_text = (response.choices[0].message.content or "").strip()
        trace.append({"iteration": MAX_AGENT_ITERATIONS + 1, "type": "forced_final"})

    if not final_text:
        raise AgentRunError("Агент не вернул финальный отчёт.")

    return final_text, charts, trace
