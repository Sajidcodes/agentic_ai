FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Streamlit config
EXPOSE 8501

# ✅ FORCE the correct entrypoint
CMD ["streamlit", "run", "fend/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
