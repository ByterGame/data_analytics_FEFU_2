import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"

MAX_FILE_SIZE_MB = 32

MAX_AGENT_ITERATIONS = 10
EXEC_TIMEOUT_SEC = 25
MAX_TOOL_OUTPUT_CHARS = 6000
MAX_CHARTS = 8
MAX_INSTRUCTION_LEN = 1000
