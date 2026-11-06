import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "")
COZE_BASE_URL = os.getenv("COZE_BASE_URL", "https://api.coze.cn")
COZE_BOT_ID = os.getenv("COZE_BOT_ID", "")
COZE_WORKFLOW_ID = os.getenv("COZE_WORKFLOW_ID", "")

WECOM_BOT_KEY = os.getenv("WECOM_BOT_KEY", "")

ERROR_LOG_ENABLED = os.getenv("ERROR_LOG_ENABLED", "true").lower() == "true"
ERROR_LOG_RETENTION_DAYS = int(os.getenv("ERROR_LOG_RETENTION_DAYS", "30"))
ERROR_LOG_CRITICAL_NOTIFY = os.getenv("ERROR_LOG_CRITICAL_NOTIFY", "true").lower() == "true"

__all__ = [
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
    "QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL",
    "COZE_API_TOKEN", "COZE_BASE_URL", "COZE_BOT_ID", "COZE_WORKFLOW_ID",
    "WECOM_BOT_KEY",
    "ERROR_LOG_ENABLED", "ERROR_LOG_RETENTION_DAYS", "ERROR_LOG_CRITICAL_NOTIFY",
]
