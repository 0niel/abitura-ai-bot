from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from config import config
from document_processor import DocumentProcessor
from logger import logger


class ChatBot:
    """Class encapsulating the chatbot functionality."""

    def __init__(self):
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
        response = self.chain.invoke(
            text,
            config={"configurable": {"search_kwargs": {"namespace": ""}}},
        )

        logger.info(f"User query: {text}")
        logger.info(f"AI response: {response}")

        # Send the response
        await context.bot.send_message(
            chat_id=update.message.chat.id,
            text=response,
            message_thread_id=update.message.message_thread_id,
            parse_mode="Markdown",
        )

        # Send feedback buttons
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data="like"),
                InlineKeyboardButton("👎", callback_data="dislike"),
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
        feedback = query.data

        # Log the feedback
        logger.info(f"User feedback: {feedback}")

        await context.bot.send_message(chat_id=config.ADMIN_CHAT_ID, text=f"Фидбэк: {feedback}")

        await query.answer("Спасибо за фидбек!")
