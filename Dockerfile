# Використовуємо офіційний образ Python як базовий
FROM python:3.10-slim

# Встановлюємо необхідні системні пакети
RUN apt-get update && apt-get install -y \
    zip \
    unzip

# Створюємо робочу директорію
WORKDIR /app

# Копіюємо requirements.txt і встановлюємо залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код додатку
COPY app.py .

# Копіюємо модель, токенайзер та датасет
COPY model/ ./model/
COPY data/ ./data/

# Відкриваємо порт для Streamlit
EXPOSE 8501

# Запускаємо додаток
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
