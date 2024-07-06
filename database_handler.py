from abc import ABC, abstractmethod
from datetime import datetime

import aiosqlite


class DatabaseHandler(ABC):
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def store_response(self, message_id: int, response: str):
        pass

    @abstractmethod
    async def update_feedback(self, message_id: int, user_id: int, feedback: str):
        pass

    @abstractmethod
    async def get_feedback(self, message_id: int):
        pass


class SQLiteHandler(DatabaseHandler):
    def __init__(self, db_path="feedback2.db"):
        self.db_path = db_path

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    response TEXT,
                    useful_count INTEGER DEFAULT 0,
                    not_useful_count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id INTEGER,
                    user_id INTEGER,
                    feedback TEXT,
                    UNIQUE(feedback_id, user_id),
                    FOREIGN KEY(feedback_id) REFERENCES feedback(id)
                )
                """
            )
            await db.commit()

    async def store_response(self, message_id: int, response: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO feedback (message_id, response) VALUES (?, ?)",
                (message_id, response),
            )
            await db.commit()

    async def update_feedback(self, message_id: int, user_id: int, feedback: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, useful_count, not_useful_count FROM feedback WHERE message_id = ?", (message_id,)
            )
            feedback_data = await cursor.fetchone()
            if not feedback_data:
                return

            feedback_id, useful_count, not_useful_count = feedback_data

            cursor = await db.execute(
                "SELECT feedback FROM user_feedback WHERE feedback_id = ? AND user_id = ?", (feedback_id, user_id)
            )
            user_feedback = await cursor.fetchone()

            if user_feedback:
                if user_feedback[0] == feedback:
                    return  # No change in feedback
                # Remove the previous feedback count
                if user_feedback[0] == "like":
                    useful_count -= 1
                else:
                    not_useful_count -= 1

            # Update the feedback count
            if feedback == "like":
                useful_count += 1
            else:
                not_useful_count += 1

            await db.execute(
                "INSERT OR REPLACE INTO user_feedback (feedback_id, user_id, feedback) VALUES (?, ?, ?)",
                (feedback_id, user_id, feedback),
            )
            await db.execute(
                "UPDATE feedback SET useful_count = ?, not_useful_count = ? WHERE id = ?",
                (useful_count, not_useful_count, feedback_id),
            )
            await db.commit()

    async def get_feedback(self, message_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT useful_count, not_useful_count, response FROM feedback WHERE message_id = ?", (message_id,)
            )
            result = await cursor.fetchone()
        return result

    async def get_overall_feedback_stats(self):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT SUM(useful_count), SUM(not_useful_count) FROM feedback")
            result = await cursor.fetchone()
        return result

    async def get_today_feedback_stats(self):
        today = datetime.now().strftime("%Y-%m-%d")
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT SUM(useful_count), SUM(not_useful_count) FROM feedback WHERE DATE(timestamp) = ?", (today,)
            )
            result = await cursor.fetchone()
        return result
