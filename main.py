import asyncio

from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from chatbot import ChatBot
from config import config
from database_handler import SQLiteHandler


async def main() -> None:
    """Starts the Telegram bot."""
    print("Starting bot...")
    # Load configuration

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).read_timeout(30).write_timeout(30).build()

    # Create a database
    db_handler = SQLiteHandler()

    # Create a ChatBot instance
    chatbot = ChatBot()
    await chatbot.initialize(db_handler)

    # Add handlers
    application.add_handler(CommandHandler("ai", chatbot.handle_ai_request))
    application.add_handler(CommandHandler("stats", chatbot.handle_stats_request))
    application.add_handler(CommandHandler("start", chatbot.start))
    application.add_handler(CallbackQueryHandler(chatbot.handle_feedback))

    # Start the bot
    async with application:
        await application.start()
        await application.updater.start_polling(pool_timeout=30)
        try:
            while True:  # THIS IS SUPPOSED TO KEEP EVENT LOOP RUNNING UNTIL CTRL-C
                await asyncio.sleep(0)
        except KeyboardInterrupt:
            await application.updater.stop()
            await application.stop()


if __name__ == "__main__":
    asyncio.run(main())
