---
title: Telegram Gemini Bot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Telegram Gemini Bot

Этот проект готов для деплоя как Docker Space на Hugging Face Spaces.

Переменные окружения, которые нужно задать в Secrets:

- `BOT_TOKEN`
- `GEMINI_API_KEY`
- `WEBHOOK_BASE_URL` — например `https://username-space-name.hf.space`
- `WEBHOOK_PATH` — можно оставить `/telegram/webhook`
- `WEBHOOK_SECRET` — любой секрет без пробелов

Локально бот запускается так же, как раньше:

```bash
python bot.py
```

Автодеплой в Hugging Face Space:

```bash
python -m pip install huggingface_hub
python deploy_hf_space.py --repo-id username/my-telegram-bot --hf-token hf_xxx
```
