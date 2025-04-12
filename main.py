import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Загрузка переменных окружения
load_dotenv()

# Конфигурация приложения
DATABASE_FILE = "complaints.db"
SENTIMENT_API_URL = "https://api.apilayer.com/sentiment/analysis"
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY")
API_TIMEOUT = 20

print(f"Using API key: {SENTIMENT_API_KEY}")

# Модели данных
class Complaint(BaseModel):
    text: str = Field(..., min_length=5, max_length=1000,
                      example="Не приходит SMS-код для входа")


class ComplaintResponse(BaseModel):
    id: int
    status: str
    sentiment: str
    category: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


# Инициализация базы данных
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sentiment TEXT,
            category TEXT DEFAULT 'other'
        )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise RuntimeError(f"Database initialization failed: {str(e)}")
    finally:
        if conn:
            conn.close()


# Локальный анализатор тональности (fallback)
def local_sentiment_analysis(text: str) -> str:
    """Анализ тональности без внешнего API"""
    negative_words = ["ненавижу", "ужас", "плохо", "кошмар", "разочарован"]
    positive_words = ["отлично", "люблю", "супер", "хорошо", "восхищаюсь"]

    text_lower = text.lower()
    if any(word in text_lower for word in negative_words):
        return "negative"
    elif any(word in text_lower for word in positive_words):
        return "positive"
    return "neutral"


# Анализ тональности с использованием API
def analyze_sentiment(text: str) -> str:
    """Анализ тональности через внешний API с fallback"""
    if not SENTIMENT_API_KEY:
        print("API key not found! Using local analyzer")
        return local_sentiment_analysis(text)

    try:
        response = requests.post(
            SENTIMENT_API_URL,
            headers={"apikey": SENTIMENT_API_KEY},
            json={"text": text},  # Важно использовать json вместо data
            timeout=API_TIMEOUT
        )
        print(f"API response: {response.status_code} {response.text}")

        if response.status_code == 200:
            result = response.json()
            sentiment = result.get("sentiment", {}).get("type", "unknown")
            return sentiment.lower() if sentiment.lower() in ["positive", "negative", "neutral"] else "unknown"

        print(f"API error: {response.status_code}. Using local analyzer")
        return local_sentiment_analysis(text)

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}. Using local analyzer")
        return local_sentiment_analysis(text)


# Определение категории жалобы
def detect_category(text: str) -> str:
    """Автоматическое определение категории"""
    text_lower = text.lower()
    tech_keywords = ["sms", "код", "техни", "приложен", "вход", "логин", "верификац"]
    payment_keywords = ["оплат", "деньг", "счет", "платеж", "перевод", "карт", "тариф"]

    if any(word in text_lower for word in tech_keywords):
        return "technical"
    elif any(word in text_lower for word in payment_keywords):
        return "payment"
    return "other"


# Обработчик жизненного цикла приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up... Initializing database")
    init_db()
    yield
    print("Shutting down...")


# Создание FastAPI приложения
app = FastAPI(
    lifespan=lifespan,
    title="Complaint Processing API",
    description="API для обработки жалоб клиентов с анализом тональности",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Эндпоинты
@app.post(
    "/complaints/",
    response_model=ComplaintResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новую жалобу",
    responses={
        400: {"model": ErrorResponse, "description": "Некорректные данные"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def create_complaint(complaint: Complaint):
    """
    Создает новую жалобу с анализом:
    - Тональности текста (positive/negative/neutral)
    - Категории (technical/payment/other)
    """
    try:
        sentiment = analyze_sentiment(complaint.text)
        category = detect_category(complaint.text)

        conn = None
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute(
                """INSERT INTO complaints (text, sentiment, category)
                   VALUES (?, ?, ?)""",
                (complaint.text, sentiment, category)
            )
            complaint_id = cursor.lastrowid
            conn.commit()

            cursor.execute(
                """SELECT id, status, sentiment, category 
                   FROM complaints WHERE id = ?""",
                (complaint_id,)
            )
            record = cursor.fetchone()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Не удалось получить созданную жалобу"
                )

            return {
                "id": record[0],
                "status": record[1],
                "sentiment": record[2],
                "category": record[3] if record[3] != "other" else None
            }

        except sqlite3.Error as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка базы данных: {str(e)}"
            )
        finally:
            if conn:
                conn.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.get(
    "/complaints/{complaint_id}",
    response_model=ComplaintResponse,
    summary="Получить информацию о жалобе",
    responses={
        404: {"model": ErrorResponse, "description": "Жалоба не найдена"},
        500: {"model": ErrorResponse, "description": "Ошибка сервера"}
    }
)
async def get_complaint(complaint_id: int):
    """Получает детали жалобы по её ID"""
    try:
        conn = None
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute(
                """SELECT id, status, sentiment, category 
                   FROM complaints WHERE id = ?""",
                (complaint_id,)
            )
            record = cursor.fetchone()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Жалоба не найдена"
                )

            return {
                "id": record[0],
                "status": record[1],
                "sentiment": record[2],
                "category": record[3] if record[3] != "other" else None
            }

        except sqlite3.Error as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ошибка базы данных: {str(e)}"
            )
        finally:
            if conn:
                conn.close()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)