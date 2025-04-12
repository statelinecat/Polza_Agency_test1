# Polza_Agency_test1
 Тестовое задание №1 от Polza Agency

### FastAPI-сервис для обработки клиентских жалоб с анализом тональности, категоризацией, проверкой на спам 
### и определением геолокации по IP с использованием публичных API.

### 📌 Функциональность
Принимает POST-запросы с текстом жалобы.

### Выполняет:
Анализ тональности (Sentiment Analysis via APILayer).
Категоризацию жалоб (технические / платежные / прочие).
Проверку на спам (Spam Check via API Ninjas, опционально).
Определение геолокации по IP (через ip-api.com).
Сохраняет жалобу в SQLite.
Возвращает JSON-ответ с данными жалобы.
Обрабатывает ошибки внешних API.

### Установка и запуск

#### 1. Клонировать репозиторий
git clone https://github.com/statelinecat/Polza_Agency_test1.git
cd Polza_Agency_test1

#### 2. Создать и активировать виртуальное окружение (опционально)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

#### 3. Установить зависимости
pip install -r requirements.txt

#### 4. Создать .env файл с API ключами
SENTIMENT_API_KEY=your_apilayer_api_key
SPAM_API_KEY=your_api_ninjas_key  # опционально

#### 5. Запустить сервер
uvicorn main:app --reload

### 🔗 Документация API
Доступна по адресу:
http://localhost:8000/docs

### 📮 Примеры запросов
#### Создание жалобы
curl -X POST http://localhost:8000/complaints/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Не приходит SMS-код"}'

#### Ответ:

{
  "id": 1,
  "status": "open",
  "sentiment": "negative",
  "category": "technical"
}

### 🧪 Тестирование
Используйте Postman или curl для тестов. Примеры выше.

### ❗ Обработка ошибок
При сбое API анализа тональности: sentiment = "unknown".
При подозрении на спам: возвращается 400 Bad Request.
При ошибках базы данных и сервера: 500 Internal Server Error.

### 📝 Лицензия
MIT License. Используйте и модифицируйте свободно.