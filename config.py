import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    ADMIN_CHAT_ID: str = os.getenv("ADMIN_CHAT_ID")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    ALLOWED_CHAT_IDS: str = os.getenv("ALLOWED_CHAT_IDS", "")
    ALLOWED_THREADS_IDS: str = os.getenv("ALLOWED_THREADS_IDS", "")
    SYSTEM_TEMPLATE: str = """Ответь на вопрос используя следующий контекст:
{context}

Вопрос: {question}

===
Ты отвечаешь в телеграме, поэтому используй Markdown форматирование. 
Не выдумывай ничего! Если информации для ответа не хватает, то ответь просто: "Я пока не могу ответить на это". 
Обязательно вставляй ссылки на документы и источнки, если они есть. 
Чтобы вставлять ссылки, которые не содержут базовый адрес (например ссылка вида '/hello-world'), добавляй https://priem.mirea.ru.
"""


config = Config()
