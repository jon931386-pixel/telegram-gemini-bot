import asyncio
import hashlib
import logging
import os
from io import BytesIO
from typing import Any

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Не задана переменная окружения {name}. "
            f"Добавь её в .env для локального запуска или в Secrets на сервере."
        )
    return value


BOT_TOKEN = require_env("BOT_TOKEN")
GEMINI_API_KEY = require_env("GEMINI_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", os.getenv("RENDER_EXTERNAL_URL", "")).strip().rstrip("/")
WEB_SERVER_HOST = os.getenv("WEB_SERVER_HOST", "0.0.0.0")
WEB_SERVER_PORT = int(os.getenv("PORT", os.getenv("WEB_SERVER_PORT", "7860")))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/telegram/webhook").strip() or "/telegram/webhook"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", hashlib.sha256(BOT_TOKEN.encode()).hexdigest())

MODEL_CANDIDATES = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

REQUEST_TIMEOUT_SECONDS = 45

TEXT_MODE_BUTTON = "💬 Текстовый запрос"
PHOTO_MODE_BUTTON = "🖼 Фото запрос"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gemini_bot")


class UserState(StatesGroup):
    waiting_text = State()
    waiting_photo = State()


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=TEXT_MODE_BUTTON),
                KeyboardButton(text=PHOTO_MODE_BUTTON),
            ]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выбери режим",
    )


class GeminiService:
    def __init__(self, api_key: str, models: tuple[str, ...], timeout_seconds: int = 45) -> None:
        self.client = genai.Client(api_key=api_key)
        self.models = models
        self.timeout_seconds = timeout_seconds

    async def ask_text(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return "Пустой запрос. Напиши текст."
        return await self._generate(prompt)

    async def ask_image(self, image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        if not image_bytes:
            return "Не удалось прочитать изображение."

        contents: list[Any] = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            (prompt or "").strip() or "Что на фото? Ответь на русском.",
        ]
        return await self._generate(contents)

    async def _generate(self, contents: Any) -> str:
        last_error: Exception | None = None

        for model_name in self.models:
            try:
                logger.info("Пробуем модель: %s", model_name)

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model_name,
                        contents=contents,
                    ),
                    timeout=self.timeout_seconds,
                )

                text = self._extract_text(response)
                if text:
                    return text

                return "Модель вернула пустой ответ. Попробуй переформулировать запрос."

            except Exception as exc:
                last_error = exc
                logger.warning("Ошибка модели %s: %s", model_name, exc, exc_info=True)

                if self._should_try_next_model(exc):
                    continue
                break

        logger.error("Все модели недоступны. Последняя ошибка: %r", last_error)
        return (
            "Все модели недоступны.\n"
            "Проверь API-ключ, доступность модели и квоту."
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) or []
                chunks: list[str] = []

                for part in parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        chunks.append(part_text.strip())

                if chunks:
                    return "\n".join(chunks)
        except Exception:
            pass

        return ""

    @staticmethod
    def _should_try_next_model(exc: Exception) -> bool:
        msg = str(exc).lower()
        retryable = (
            "429",
            "quota",
            "rate limit",
            "resource exhausted",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "deadline exceeded",
            "404",
            "not found",
            "model",
        )
        return any(marker in msg for marker in retryable)


gemini_service = GeminiService(
    api_key=GEMINI_API_KEY,
    models=MODEL_CANDIDATES,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


def webhook_url() -> str:
    return f"{WEBHOOK_BASE_URL}{WEBHOOK_PATH}"


def running_in_webhook_mode() -> bool:
    return bool(WEBHOOK_BASE_URL)


async def configure_webhook_with_retry(attempts: int = 10, delay_seconds: int = 6) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            await bot.set_webhook(
                url=webhook_url(),
                secret_token=WEBHOOK_SECRET,
                allowed_updates=dp.resolve_used_update_types(),
            )
            logger.info("Webhook установлен: %s", webhook_url())
            return True
        except Exception as exc:
            logger.warning(
                "Не удалось установить webhook, попытка %s/%s: %s",
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(delay_seconds)

    logger.error("Webhook не установлен автоматически. Бот продолжает работу и ждёт повторной настройки.")
    return False


async def disable_webhook_for_polling() -> None:
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        logger.info("Webhook отключён, бот работает через polling.")
    except Exception as exc:
        logger.warning("Не удалось отключить webhook перед polling: %s", exc)


@dp.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Привет! Выбери режим:", reply_markup=main_menu())


@dp.message(Command("cancel"))
async def cancel_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено. Выбери режим:", reply_markup=main_menu())


@dp.message(F.text == TEXT_MODE_BUTTON)
async def text_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_text)
    await message.answer("Напиши свой вопрос.")


@dp.message(F.text == PHOTO_MODE_BUTTON)
async def photo_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_photo)
    await message.answer("Пришли фото, можно с подписью.")


@dp.message(StateFilter(UserState.waiting_text), F.text)
async def process_text_handler(message: Message) -> None:
    if not message.text or message.text in {TEXT_MODE_BUTTON, PHOTO_MODE_BUTTON}:
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    reply = await gemini_service.ask_text(message.text)
    await message.answer(reply)


@dp.message(StateFilter(UserState.waiting_text))
async def process_wrong_text_input(message: Message) -> None:
    await message.answer("Сейчас активен текстовый режим. Напиши сообщение текстом.")


@dp.message(StateFilter(UserState.waiting_photo), F.photo)
async def process_photo_handler(message: Message) -> None:
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        photo = message.photo[-1]
        file_buffer = await message.bot.download(photo)

        if not isinstance(file_buffer, BytesIO):
            await message.answer("Не удалось скачать фото.")
            return

        image_bytes = file_buffer.getvalue()
        caption = (message.caption or "").strip() or "Что на фото? Ответь на русском."

        reply = await gemini_service.ask_image(
            image_bytes=image_bytes,
            prompt=caption,
            mime_type="image/jpeg",
        )
        await message.answer(reply)

    except Exception as exc:
        logger.exception("Ошибка обработки фото: %s", exc)
        await message.answer("Ошибка при обработке фото. Попробуй ещё раз.")


@dp.message(StateFilter(UserState.waiting_photo))
async def process_wrong_photo_input(message: Message) -> None:
    await message.answer("Сейчас активен режим фото. Пришли изображение.")


@dp.message()
async def fallback_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()

    if current_state is None:
        await message.answer("Выбери режим:", reply_markup=main_menu())
        return

    await message.answer("Не понял запрос. Используй кнопки или /cancel.")


async def healthcheck(_: web.Request) -> web.Response:
    mode = "webhook" if running_in_webhook_mode() else "polling"
    return web.json_response({"ok": True, "mode": mode})


async def telegram_check(_: web.Request) -> web.Response:
    try:
        me = await asyncio.wait_for(bot.get_me(request_timeout=8), timeout=10)
        return web.json_response(
            {
                "ok": True,
                "telegram_api": True,
                "bot_username": me.username,
                "bot_id": me.id,
            }
        )
    except Exception as exc:
        return web.json_response(
            {
                "ok": False,
                "telegram_api": False,
                "error": str(exc),
            },
            status=503,
        )


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", healthcheck)
    app.router.add_get("/healthz", healthcheck)
    app.router.add_get("/telegram-check", telegram_check)

    SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=WEBHOOK_SECRET,
    ).register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)
    return app


async def run_webhook() -> None:
    app = create_web_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)
    await site.start()

    logger.info("Бот запущен в webhook-режиме на %s:%s", WEB_SERVER_HOST, WEB_SERVER_PORT)
    logger.info("Webhook path: %s", WEBHOOK_PATH)
    await configure_webhook_with_retry()

    await asyncio.Event().wait()


async def run_polling() -> None:
    logger.info("Бот запускается в polling-режиме...")
    await disable_webhook_for_polling()
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


async def main() -> None:
    if running_in_webhook_mode():
        await run_webhook()
    else:
        await run_polling()


if __name__ == "__main__":
    asyncio.run(main())
