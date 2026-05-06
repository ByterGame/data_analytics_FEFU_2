import re

from .config import MAX_INSTRUCTION_LEN

_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|previous|prior|above)",
    r"забудь\s+(все|всё|предыдущие|выше|инструкции)",
    r"игнорируй\s+(все|всё|предыдущие|выше|инструкции|правила)",
    r"проигнорируй\s+(все|всё|предыдущие|выше|инструкции)",
    r"new\s+(system|instruction)s?\s*:",
    r"system\s*:\s*",
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    r"show\s+me\s+(your|the)\s+(system\s+)?prompt",
    r"раскрой\s+системн",
    r"покажи\s+системн",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(different|another|new)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"prompt\s*injection",
    r"override\s+(your|the)\s+(instructions|rules|system)",
    r"</?\s*(system|user|assistant|tool)\s*>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def sanitize_instruction(text: str) -> tuple[str, bool]:
    """Возвращает (cleaned, suspicious). Cleaned обрезан до MAX_INSTRUCTION_LEN."""
    if not text:
        return "", False

    cleaned = text.strip()
    if len(cleaned) > MAX_INSTRUCTION_LEN:
        cleaned = cleaned[:MAX_INSTRUCTION_LEN]

    suspicious = any(p.search(cleaned) for p in _COMPILED)
    return cleaned, suspicious
