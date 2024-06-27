from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from config import config
from database_handler import DatabaseHandler
from document_processor import DocumentProcessor
from logger import logger


class ChatBot:
    """Class encapsulating the chatbot functionality."""

    async def initialize(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        await self.db_handler.initialize()
        self.prompt = self._create_prompt()
        self.llm = self._create_llm()
        self.vectorstore = self._create_vectorstore()
        self.retriever = self._create_retriever()
        self.chain = self._create_chain()

    def _create_prompt(self) -> ChatPromptTemplate:
        """Creates and returns a ChatPromptTemplate."""
        return ChatPromptTemplate.from_template(config.SYSTEM_TEMPLATE)

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Creates and returns a ChatGoogleGenerativeAI instance."""
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
        )

    def _create_vectorstore(self) -> Chroma:
        """Creates and returns a Chroma vectorstore instance."""
        return Chroma(
            persist_directory=DocumentProcessor.PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings(
                retry_max_seconds=120,
                show_progress_bar=True,
            ),
        )

    def _create_retriever(self) -> ConfigurableField:
        """Creates and returns a ConfigurableField retriever."""
        retriever = self.vectorstore.as_retriever(k=4)
        return retriever.configurable_fields()

    def _create_chain(self):
        """Creates and returns the processing chain."""
        return (
            {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
        )

    async def handle_ai_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the AI request from Telegram."""
        text = update.message.text[4:].strip()
        if not text:
            return

        # Send typing action
        await context.bot.send_chat_action(
            chat_id=update.message.chat.id,
            action="typing",
            message_thread_id=update.message.message_thread_id,
        )

        # Invoke the chain
        response = await self.chain.ainvoke(
            text,
            config={"configurable": {"search_kwargs": {"namespace": ""}}},
        )

        logger.info(f"User query: {text}")
        logger.info(f"AI response: {response}")

        # Send the response
        message = await context.bot.send_message(
            chat_id=update.message.chat.id,
            text=response,
            message_thread_id=update.message.message_thread_id,
            parse_mode="Markdown",
        )

        # Store response in database
        await self.db_handler.store_response(message.message_id, response)

        # Send feedback buttons
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data=f"like:{message.message_id}"),
                InlineKeyboardButton("👎", callback_data=f"dislike:{message.message_id}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=update.message.chat.id,
            text="Этот ответ был полезен?",
            reply_markup=reply_markup,
            message_thread_id=update.message.message_thread_id,
        )

        await context.bot.send_chat_action(
            chat_id=update.message.chat.id,
            action="cancel",
            message_thread_id=update.message.message_thread_id,
        )

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the feedback from the user."""
        query = update.callback_query
        try:
            feedback, message_id = query.data.split(":")
            message_id = int(message_id)
        except ValueError:
            logger.error(f"Invalid callback data: {query.data}")
            await query.answer("Некорректные данные фидбэка.")
            return

        user_id = query.from_user.id

        await self.db_handler.update_feedback(message_id, user_id, feedback)
        useful_count, not_useful_count, response = await self.db_handler.get_feedback(message_id)

        feedback_message = f"Этот ответ был полезен?\n\n👍 {useful_count} | 👎 {not_useful_count}"

        await query.edit_message_text(text=feedback_message, reply_markup=query.message.reply_markup)

        logger.info(f"User feedback: {feedback} for message_id {message_id}")

        await context.bot.send_message(chat_id=config.ADMIN_CHAT_ID, text=f"Фидбэк: {feedback}\nОтвет: {response}")

        await query.answer("Спасибо за фидбэк!")
