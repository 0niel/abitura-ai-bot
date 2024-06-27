from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler

from chatbot import ChatBot
from config import config


def main() -> None:
    """Starts the Telegram bot."""
    print("Starting bot...")
    # Load configuration

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Create a ChatBot instance
    chatbot = ChatBot()

    # Add handlers
    application.add_handler(CommandHandler("ai", chatbot.handle_ai_request))
    application.add_handler(CallbackQueryHandler(chatbot.handle_feedback))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
