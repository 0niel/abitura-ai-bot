from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters, Update
from telegram.ext import ContextTypes

from config import config
from database_handler import DatabaseHandler
from document_processor import DocumentProcessor
from logger import logger


class Question(BaseModel):
    __root__: str


class PromptCreator:
    @staticmethod
    def create_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(config.SYSTEM_TEMPLATE)


class LLMProvider:
    @staticmethod
    def create_llm() -> ChatGoogleGenerativeAI:
        return ChatOpenAI(
            temperature=0.2,
            max_retries=16,
            timeout=120,
            model="gpt-4o",
        )


class VectorStoreProvider:
    @staticmethod
    def create_vectorstore() -> Chroma:
        return Chroma(
            persist_directory=DocumentProcessor.PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings(
                retry_max_seconds=120,
                show_progress_bar=True,
                max_retries=10,
            ),
        )


class RetrieverProvider:
    @staticmethod
    def create_retriever(vectorstore: Chroma, llm: ChatGoogleGenerativeAI) -> MultiQueryRetriever:
        return MultiQueryRetriever.from_llm(vectorstore.as_retriever(k=5), llm)


class ProcessingChain:
    @staticmethod
    def create_chain(prompt: ChatPromptTemplate, llm: ChatGoogleGenerativeAI, retriever: MultiQueryRetriever):
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.with_types(input_type=Question)


class ChatBot:
    def __init__(self):
        self.db_handler = None
        self.prompt = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.allowed_chat_ids = [int(chat_id) for chat_id in config.ALLOWED_CHAT_IDS.split(",") if chat_id]
        self.allowed_thread_ids = [int(thread_id) for thread_id in config.ALLOWED_THREADS_IDS.split(",") if thread_id]

    async def initialize(self, db_handler: DatabaseHandler):
        self.db_handler = db_handler
        await self.db_handler.initialize()
        self.prompt = PromptCreator.create_prompt()
        self.llm = LLMProvider.create_llm()
        self.vectorstore = VectorStoreProvider.create_vectorstore()
        self.retriever = RetrieverProvider.create_retriever(self.vectorstore, self.llm)
        self.chain = ProcessingChain.create_chain(self.prompt, self.llm, self.retriever)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.message.chat.id
        thread_id = update.message.message_thread_id

        if not self._is_allowed_chat(chat_id, thread_id):
            await self._send_restricted_access_message(context, chat_id, thread_id)
            return

        user_first_name = update.message.from_user.first_name

        welcome_message_1 = (
            f"👋 *Привет, {user_first_name}!* \n\n"
            "Я ваш AI-бот, _помощник абитуриентов_.\n"
            "Готов помочь вам с любой информацией о поступлении. 🎓"
        )

        welcome_message_2 = (
            "Вот несколько примеров вопросов, которые вы можете задать:\n"
            "• *Какие документы нужны для поступления?* 📄\n"
            "• *Какие вступительные экзамены нужно сдать?* 📝\n"
            "• *Какой проходной балл в этом году?* 🎓\n"
            "• *Как подать заявление на общежитие?* 🏠\n\n"
            "Чтобы задать вопрос, просто напишите его после команды /ai.\n"
            "_Например_: `/ai Какие документы нужны для поступления?`"
        )

        await context.bot.send_message(
            chat_id=chat_id,
            text=welcome_message_1,
            message_thread_id=thread_id,
            parse_mode="Markdown",
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text=welcome_message_2,
            message_thread_id=thread_id,
            parse_mode="Markdown",
        )

    async def handle_ai_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.message.chat.id
        thread_id = update.message.message_thread_id

        if not self._is_allowed_chat(chat_id, thread_id):
            await self._send_restricted_access_message(context, chat_id, thread_id)
            return

        text = update.message.text[4:].strip()
        if not text:
            return

        await self._send_typing_action(context, chat_id, thread_id)

        response = await self.chain.ainvoke(
            text,
            config={"configurable": {"search_kwargs": {"namespace": ""}}},
        )

        logger.info(f"User query: {text}")
        logger.info(f"AI response: {response}")

        reply_to_message_id = update.message.message_id
        message = await self._send_response_message(context, chat_id, thread_id, response, reply_to_message_id)

        await self.db_handler.store_response(message.message_id, response)

        await self._send_feedback_buttons(context, chat_id, thread_id, message.message_id)

        await context.bot.send_chat_action(
            chat_id=chat_id,
            action="cancel",
            message_thread_id=thread_id,
        )

    def _is_allowed_chat(self, chat_id: int, thread_id: int) -> bool:
        return (chat_id in self.allowed_chat_ids or not self.allowed_chat_ids) and (
            thread_id in self.allowed_thread_ids or not self.allowed_thread_ids
        )

    async def _send_restricted_access_message(self, context, chat_id, thread_id):
        await context.bot.send_message(
            chat_id=chat_id,
            text="Этот бот может общаться только в определённых чатах и тредах.",
            message_thread_id=thread_id,
            parse_mode="Markdown",
        )

    async def _send_typing_action(self, context, chat_id, thread_id):
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action="typing",
            message_thread_id=thread_id,
        )

    async def _send_response_message(self, context, chat_id, thread_id, response, reply_to_message_id):
        return await context.bot.send_message(
            chat_id=chat_id,
            text=response,
            message_thread_id=thread_id,
            reply_parameters=ReplyParameters(message_id=reply_to_message_id),
            parse_mode="Markdown",
        )

    async def _send_feedback_buttons(self, context, chat_id, thread_id, message_id):
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data=f"like:{message_id}"),
                InlineKeyboardButton("👎", callback_data=f"dislike:{message_id}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Этот ответ был полезен?",
            reply_markup=reply_markup,
            message_thread_id=thread_id,
        )

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        feedback, message_id = self._parse_feedback(query)
        if feedback is None:
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

    def _parse_feedback(self, query):
        try:
            feedback, message_id = query.data.split(":")
            message_id = int(message_id)
            return feedback, message_id
        except ValueError:
            logger.error(f"Invalid callback data: {query.data}")
            return None, None

    async def handle_stats_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.message.chat.id
        thread_id = update.message.message_thread_id

        if not self._is_allowed_chat(chat_id, thread_id):
            await self._send_restricted_access_message(context, chat_id, thread_id)
            return

        overall_stats = await self.db_handler.get_overall_feedback_stats()
        today_stats = await self.db_handler.get_today_feedback_stats()

        overall_useful, overall_not_useful = overall_stats or (0, 0)
        today_useful, today_not_useful = today_stats if today_stats else (0, 0)

        response = (
            f"📊 *Статистика ответов*\n\n"
            f"Всего полезных: {overall_useful}\n"
            f"Всего бесполезных: {overall_not_useful}\n\n"
            f"Полезных за сегодня: {today_useful}\n"
            f"Бесполезных за сегодня: {today_not_useful}"
        )

        await context.bot.send_message(
            chat_id=chat_id,
            text=response,
            message_thread_id=thread_id,
            parse_mode="Markdown",
        )
