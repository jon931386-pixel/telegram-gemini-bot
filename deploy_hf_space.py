import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from huggingface_hub import HfApi, SpaceHardware
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Не найден пакет huggingface_hub. Установи его командой: "
        "python -m pip install huggingface_hub"
    ) from exc


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


def require_value(name: str, value: str | None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise SystemExit(f"Не задано значение {name}.")
    return cleaned


def build_space_url(repo_id: str) -> str:
    return f"https://{repo_id.replace('/', '-').lower()}.hf.space"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создать или обновить Hugging Face Space для Telegram-бота."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Идентификатор Space в формате username/space-name",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        help="Токен Hugging Face. Можно не передавать, если он уже есть в переменной окружения HF_TOKEN.",
    )
    parser.add_argument(
        "--bot-token",
        default=os.getenv("BOT_TOKEN"),
        help="Токен Telegram-бота. По умолчанию берётся из .env",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.getenv("GEMINI_API_KEY"),
        help="API-ключ Gemini. По умолчанию берётся из .env",
    )
    parser.add_argument(
        "--webhook-path",
        default=os.getenv("WEBHOOK_PATH", "/telegram/webhook"),
        help="Путь webhook внутри сервиса. По умолчанию /telegram/webhook",
    )
    parser.add_argument(
        "--webhook-secret",
        default=os.getenv("WEBHOOK_SECRET", ""),
        help="Секрет webhook. Если не передан, бот сам вычислит его из BOT_TOKEN.",
    )
    args = parser.parse_args()

    hf_token = require_value("HF token", args.hf_token)
    repo_id = require_value("repo-id", args.repo_id)
    bot_token = require_value("BOT_TOKEN", args.bot_token)
    gemini_api_key = require_value("GEMINI_API_KEY", args.gemini_api_key)
    webhook_path = require_value("WEBHOOK_PATH", args.webhook_path)
    webhook_base_url = build_space_url(repo_id)

    api = HfApi(token=hf_token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        space_hardware=SpaceHardware.CPU_BASIC,
        exist_ok=True,
    )

    api.add_space_secret(repo_id=repo_id, key="BOT_TOKEN", value=bot_token)
    api.add_space_secret(repo_id=repo_id, key="GEMINI_API_KEY", value=gemini_api_key)

    if args.webhook_secret.strip():
        api.add_space_secret(
            repo_id=repo_id,
            key="WEBHOOK_SECRET",
            value=args.webhook_secret.strip(),
        )

    api.add_space_variable(repo_id=repo_id, key="WEBHOOK_BASE_URL", value=webhook_base_url)
    api.add_space_variable(repo_id=repo_id, key="WEBHOOK_PATH", value=webhook_path)
    api.add_space_variable(repo_id=repo_id, key="PORT", value="7860")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=PROJECT_DIR,
        commit_message="Deploy Telegram Gemini bot",
        ignore_patterns=[
            ".env",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".venv",
            "venv",
            "*.log",
        ],
    )

    print("Готово.")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"Публичный URL: {webhook_base_url}")
    print("После сборки отправь боту /start в Telegram.")


if __name__ == "__main__":
    main()
