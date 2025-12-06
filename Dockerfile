# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for chromadb, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*


# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "fend/f1.py", "--server.port=8501", "--server.address=0.0.0.0"]
