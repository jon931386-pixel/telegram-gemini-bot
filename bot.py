import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
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
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "bot_memory.sqlite3")

MODEL_CANDIDATES = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_MEMORY_MESSAGE_CHARS = int(os.getenv("MAX_MEMORY_MESSAGE_CHARS", "2500"))
IMAGE_CONTEXT_TTL_SECONDS = int(os.getenv("IMAGE_CONTEXT_TTL_SECONDS", "3600"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1400"))
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_SAFE_CHUNK_SIZE = 3800

TEXT_MODE_BUTTON = "💬 Текстовый запрос"
PHOTO_MODE_BUTTON = "🖼 Фото запрос"

SYSTEM_INSTRUCTION = (
    "Ты дружелюбный русскоязычный помощник в Telegram. "
    "Отвечай естественно и по делу, без фраз вроде 'я языковая модель', если пользователь об этом не спрашивал. "
    "Учитывай недавнюю историю диалога. "
    "Если пользователь пишет короткое уточнение вроде 'ну?', 'продолжай', 'реши их', "
    "связывай его с предыдущим контекстом, а не отвечай как на новый независимый вопрос. "
    "Если пользователь явно начал новую самостоятельную тему, не тащи в ответ прошлую задачу, прошлое фото и старый контекст. "
    "Если пользователь уже присылал изображение и потом даёт текстовое уточнение в контексте фото, "
    "считай, что он имеет в виду последнее изображение. "
    "Если ответ получается длинным, сначала дай полезную суть, потом детали. "
    "Отвечай на русском языке, если пользователь не просит иное."
)

WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")

STOP_WORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко", "у", "о", "об", "от", "за", "из",
    "под", "над", "до", "для", "при", "не", "ни", "а", "но", "или", "ли", "же", "бы",
    "это", "этот", "эта", "эти", "то", "та", "те", "так", "как", "что", "чтобы", "если",
    "уже", "ещё", "еще", "там", "тут", "здесь", "меня", "мне", "мой", "моя", "мои", "мое",
    "твой", "твоя", "твои", "тебе", "его", "ее", "её", "их", "они", "она", "оно", "он",
    "мы", "вы", "ты", "я", "ну", "да", "нет", "просто", "надо", "нужно", "можно", "давай",
}

CONTINUATION_PHRASES = {
    "ну", "ну?", "продолжай", "продолжи", "дальше", "еще", "ещё", "что дальше",
    "реши", "реши их", "решай", "ответь", "объясни", "поясни", "распиши",
    "расскажи подробнее", "подробнее", "что там", "и что", "и дальше",
    "как меня зовут", "что я говорил", "что было раньше", "повтори",
    "сделай короче", "сделай кратко", "продолжай решение",
}

PHOTO_CONTINUATION_PHRASES = {
    "реши", "реши их", "что на фото", "прочитай", "разбери", "объясни задание",
    "ответь по фото", "решай дальше", "дочитай", "что тут написано",
}

COMMON_WORD_ENDINGS = (
    "иями", "ями", "ами", "его", "ого", "ему", "ому", "иях", "ах", "ях",
    "ией", "ей", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие",
    "ую", "юю", "ам", "ям", "ом", "ем", "ов", "ев", "а", "я", "ы", "и", "е", "у", "ю", "о",
)


@dataclass
class PromptRoutingDecision:
    reset_context: bool
    use_image_context: bool


def normalize_prompt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def extract_keywords(text: str) -> set[str]:
    words = {
        match.group(0).lower()
        for match in WORD_RE.finditer(text or "")
    }
    return {
        stem_word(word)
        for word in words
        if len(word) >= 3 and word not in STOP_WORDS
    }


def stem_word(word: str) -> str:
    normalized = word.lower()
    for ending in COMMON_WORD_ENDINGS:
        if len(normalized) > len(ending) + 3 and normalized.endswith(ending):
            return normalized[: -len(ending)]
    return normalized


def contains_any_phrase(text: str, phrases: set[str]) -> bool:
    normalized = normalize_prompt(text)
    return any(phrase in normalized for phrase in phrases)


def is_short_followup(text: str) -> bool:
    normalized = normalize_prompt(text)
    return len(normalized) <= 40 or len(extract_keywords(normalized)) <= 2


def recent_topic_overlap(prompt: str, recent_texts: list[str]) -> float:
    prompt_keywords = extract_keywords(prompt)
    recent_keywords = extract_keywords(" ".join(recent_texts))

    if not prompt_keywords or not recent_keywords:
        return 0.0

    shared = prompt_keywords & recent_keywords
    return len(shared) / max(1, len(prompt_keywords))


def decide_prompt_routing(prompt: str, recent_texts: list[str], has_image_context: bool) -> PromptRoutingDecision:
    normalized = normalize_prompt(prompt)

    if not normalized:
        return PromptRoutingDecision(reset_context=False, use_image_context=False)

    if contains_any_phrase(normalized, CONTINUATION_PHRASES):
        return PromptRoutingDecision(
            reset_context=False,
            use_image_context=has_image_context and contains_any_phrase(normalized, PHOTO_CONTINUATION_PHRASES),
        )

    overlap = recent_topic_overlap(normalized, recent_texts)
    standalone_prompt = len(normalized) >= 20 or len(extract_keywords(normalized)) >= 3
    clear_topic_shift = standalone_prompt and overlap < 0.2

    if clear_topic_shift:
        return PromptRoutingDecision(reset_context=True, use_image_context=False)

    if has_image_context and contains_any_phrase(normalized, PHOTO_CONTINUATION_PHRASES):
        return PromptRoutingDecision(reset_context=False, use_image_context=True)

    if has_image_context and is_short_followup(normalized) and overlap >= 0.2:
        return PromptRoutingDecision(reset_context=False, use_image_context=True)

    return PromptRoutingDecision(reset_context=False, use_image_context=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("telegram_gemini_bot")


class UserState(StatesGroup):
    waiting_text = State()
    waiting_photo = State()


@dataclass
class ImageContext:
    image_bytes: bytes
    mime_type: str
    saved_at: float


class ConversationMemory:
    def __init__(self, db_path: str, max_messages: int, max_message_chars: int) -> None:
        self.db_path = db_path
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars
        self._lock = threading.Lock()
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_db(self) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_chat_id_id
                    ON conversation_messages (chat_id, id)
                    """
                )

    def _clip_for_memory(self, text: str) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= self.max_message_chars:
            return cleaned
        return cleaned[: self.max_message_chars].rstrip() + "\n\n[Обрезано для памяти]"

    def _prune(self, connection: sqlite3.Connection, chat_id: int) -> None:
        rows = connection.execute(
            """
            SELECT id
            FROM conversation_messages
            WHERE chat_id = ?
            ORDER BY id DESC
            """,
            (chat_id,),
        ).fetchall()

        stale_rows = rows[self.max_messages :]
        if stale_rows:
            connection.executemany(
                "DELETE FROM conversation_messages WHERE id = ?",
                [(row["id"],) for row in stale_rows],
            )

    def _add_exchange_sync(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        clipped_user = self._clip_for_memory(user_text)
        clipped_assistant = self._clip_for_memory(assistant_text)

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO conversation_messages (chat_id, role, content) VALUES (?, ?, ?)",
                    (chat_id, "user", clipped_user),
                )
                connection.execute(
                    "INSERT INTO conversation_messages (chat_id, role, content) VALUES (?, ?, ?)",
                    (chat_id, "assistant", clipped_assistant),
                )
                self._prune(connection, chat_id)

    async def add_exchange(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        await asyncio.to_thread(self._add_exchange_sync, chat_id, user_text, assistant_text)

    def _get_contents_sync(self, chat_id: int) -> list[types.Content]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM conversation_messages
                    WHERE chat_id = ?
                    ORDER BY id ASC
                    """,
                    (chat_id,),
                ).fetchall()

        contents: list[types.Content] = []
        for row in rows:
            part = types.Part(text=row["content"])
            if row["role"] == "assistant":
                contents.append(types.ModelContent(parts=[part]))
            else:
                contents.append(types.UserContent(parts=[part]))
        return contents

    async def get_contents(self, chat_id: int) -> list[types.Content]:
        return await asyncio.to_thread(self._get_contents_sync, chat_id)

    def _get_recent_texts_sync(self, chat_id: int, limit: int) -> list[str]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT content
                    FROM conversation_messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                ).fetchall()

        return [row["content"] for row in reversed(rows)]

    async def get_recent_texts(self, chat_id: int, limit: int = 6) -> list[str]:
        return await asyncio.to_thread(self._get_recent_texts_sync, chat_id, limit)

    def _clear_chat_sync(self, chat_id: int) -> None:
        with self._lock:
            with self._connect() as connection:
                connection.execute("DELETE FROM conversation_messages WHERE chat_id = ?", (chat_id,))

    async def clear_chat(self, chat_id: int) -> None:
        await asyncio.to_thread(self._clear_chat_sync, chat_id)


class ImageSessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[int, ImageContext] = {}

    def remember(self, chat_id: int, image_bytes: bytes, mime_type: str) -> None:
        self._items[chat_id] = ImageContext(
            image_bytes=image_bytes,
            mime_type=mime_type,
            saved_at=time.time(),
        )

    def get(self, chat_id: int) -> ImageContext | None:
        context = self._items.get(chat_id)
        if context is None:
            return None

        if time.time() - context.saved_at > self.ttl_seconds:
            self._items.pop(chat_id, None)
            return None

        return context

    def clear(self, chat_id: int) -> None:
        self._items.pop(chat_id, None)


class RepeatingChatAction:
    def __init__(self, bot: Bot, chat_id: int, action: ChatAction = ChatAction.TYPING, interval: float = 4.0) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.action = action
        self.interval = interval
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            await self.bot.send_chat_action(self.chat_id, self.action)
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task


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


def split_text_for_telegram(text: str, chunk_size: int = TELEGRAM_SAFE_CHUNK_SIZE) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return ["Не удалось сформировать ответ."]

    chunks: list[str] = []
    remaining = normalized

    while len(remaining) > TELEGRAM_TEXT_LIMIT:
        split_at = remaining.rfind("\n\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind(". ", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind(" ", 0, chunk_size)
        if split_at <= 0:
            split_at = chunk_size

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:chunk_size]
            split_at = chunk_size

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


async def answer_in_chunks(message: Message, text: str) -> None:
    for chunk in split_text_for_telegram(text):
        await message.answer(chunk)


class GeminiService:
    def __init__(
        self,
        api_key: str,
        models: tuple[str, ...],
        memory: ConversationMemory,
        timeout_seconds: int = 90,
        max_output_tokens: int = 1400,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.models = models
        self.memory = memory
        self.timeout_seconds = timeout_seconds
        self.generation_config = types.GenerateContentConfig(
            systemInstruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            maxOutputTokens=max_output_tokens,
        )

    async def ask_text(self, chat_id: int, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return "Пустой запрос. Напиши текст."

        logger.info("Текстовый запрос chat_id=%s chars=%s", chat_id, len(prompt))
        contents = await self.memory.get_contents(chat_id)
        contents.append(types.UserContent(parts=[types.Part(text=prompt)]))

        reply = await self._generate(contents)
        await self.memory.add_exchange(chat_id, prompt, reply)
        return reply

    async def ask_image(self, chat_id: int, image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg") -> str:
        if not image_bytes:
            return "Не удалось прочитать изображение."

        prompt = (prompt or "").strip() or "Что на фото? Ответь на русском."
        logger.info("Фото-запрос chat_id=%s prompt_chars=%s image_bytes=%s", chat_id, len(prompt), len(image_bytes))

        contents = await self.memory.get_contents(chat_id)
        contents.append(
            types.UserContent(
                parts=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    types.Part(text=prompt),
                ]
            )
        )

        reply = await self._generate(contents)
        await self.memory.add_exchange(chat_id, f"[Фото] {prompt}", reply)
        return reply

    async def _generate(self, contents: list[types.Content]) -> str:
        last_error: Exception | None = None

        for model_name in self.models:
            try:
                logger.info("Пробуем модель: %s", model_name)

                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model_name,
                        contents=contents,
                        config=self.generation_config,
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

        error_message = str(last_error or "").lower()
        if "timeout" in error_message or "deadline exceeded" in error_message:
            return (
                "Ответ готовился слишком долго и запрос оборвался по времени. "
                "Попробуй сократить запрос или разбить его на части."
            )

        return (
            "Сейчас не получилось получить ответ от модели. "
            "Попробуй ещё раз через несколько секунд."
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
            "internal",
            "unavailable",
            "404",
            "not found",
            "model",
        )
        return any(marker in msg for marker in retryable)


conversation_memory = ConversationMemory(
    db_path=MEMORY_DB_PATH,
    max_messages=MAX_HISTORY_MESSAGES,
    max_message_chars=MAX_MEMORY_MESSAGE_CHARS,
)
image_context_store = ImageSessionStore(ttl_seconds=IMAGE_CONTEXT_TTL_SECONDS)

gemini_service = GeminiService(
    api_key=GEMINI_API_KEY,
    models=MODEL_CANDIDATES,
    memory=conversation_memory,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    max_output_tokens=MAX_OUTPUT_TOKENS,
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


async def process_text_prompt(message: Message, state: FSMContext, prompt: str) -> None:
    await state.set_state(UserState.waiting_text)
    async with RepeatingChatAction(message.bot, message.chat.id):
        reply = await gemini_service.ask_text(chat_id=message.chat.id, prompt=prompt)
    await answer_in_chunks(message, reply)


async def process_photo_prompt(
    message: Message,
    state: FSMContext,
    prompt: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> None:
    await state.set_state(UserState.waiting_photo)
    async with RepeatingChatAction(message.bot, message.chat.id):
        reply = await gemini_service.ask_image(
            chat_id=message.chat.id,
            image_bytes=image_bytes,
            prompt=prompt,
            mime_type=mime_type,
        )
    await answer_in_chunks(message, reply)


@dp.message(CommandStart())
async def start_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Привет! Можешь писать текстом или присылать фото. "
        "Кнопки ниже просто помогают переключать режимы. "
        "Чтобы очистить память диалога, используй /new.",
        reply_markup=main_menu(),
    )


@dp.message(Command("cancel"))
async def cancel_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Режим сброшен. Память диалога сохранена. "
        "Если хочешь начать с нуля, используй /new.",
        reply_markup=main_menu(),
    )


@dp.message(Command(commands=["new", "reset", "clear"]))
async def new_dialog_handler(message: Message, state: FSMContext) -> None:
    await state.clear()
    await conversation_memory.clear_chat(message.chat.id)
    image_context_store.clear(message.chat.id)
    await message.answer(
        "Память диалога очищена. Начинаем заново.",
        reply_markup=main_menu(),
    )


@dp.message(F.text == TEXT_MODE_BUTTON)
async def text_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_text)
    await message.answer("Текстовый режим включён. Напиши свой вопрос.")


@dp.message(F.text == PHOTO_MODE_BUTTON)
async def photo_mode_handler(message: Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_photo)
    await message.answer(
        "Фото-режим включён. Пришли фото или уточнение по последнему фото."
    )


@dp.message(F.photo)
async def photo_message_handler(message: Message, state: FSMContext) -> None:
    try:
        photo = message.photo[-1]
        file_buffer = await message.bot.download(photo)

        if not isinstance(file_buffer, BytesIO):
            await message.answer("Не удалось скачать фото.")
            return

        image_bytes = file_buffer.getvalue()
        image_context_store.remember(message.chat.id, image_bytes, "image/jpeg")

        prompt = (message.caption or "").strip() or "Что на фото? Ответь на русском."
        await process_photo_prompt(
            message=message,
            state=state,
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type="image/jpeg",
        )
    except Exception as exc:
        logger.exception("Ошибка обработки фото: %s", exc)
        await message.answer("Ошибка при обработке фото. Попробуй ещё раз.")


@dp.message(F.text)
async def text_message_handler(message: Message, state: FSMContext) -> None:
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer("Напиши текстовый запрос или пришли фото.")
        return

    current_state = await state.get_state()
    recent_texts = await conversation_memory.get_recent_texts(message.chat.id, limit=6)
    image_context = image_context_store.get(message.chat.id)
    routing = decide_prompt_routing(
        prompt=prompt,
        recent_texts=recent_texts,
        has_image_context=image_context is not None,
    )

    try:
        if routing.reset_context:
            logger.info("Обнаружена новая тема chat_id=%s, сбрасываем старый контекст.", message.chat.id)
            await conversation_memory.clear_chat(message.chat.id)
            image_context_store.clear(message.chat.id)
            await state.set_state(UserState.waiting_text)

        elif current_state == UserState.waiting_photo.state and image_context is not None and routing.use_image_context:
            logger.info("Используем последнее фото как контекст chat_id=%s.", message.chat.id)
            if image_context is not None:
                await process_photo_prompt(
                    message=message,
                    state=state,
                    prompt=prompt,
                    image_bytes=image_context.image_bytes,
                    mime_type=image_context.mime_type,
                )
                return

        await process_text_prompt(message=message, state=state, prompt=prompt)
    except Exception as exc:
        logger.exception("Ошибка обработки текстового запроса: %s", exc)
        await message.answer(
            "Не получилось отправить ответ. Попробуй ещё раз, "
            "а если запрос очень большой — разбей его на части."
        )


@dp.message()
async def fallback_handler(message: Message) -> None:
    await message.answer(
        "Я понимаю текст и фото. Напиши сообщение или пришли изображение.",
        reply_markup=main_menu(),
    )


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
