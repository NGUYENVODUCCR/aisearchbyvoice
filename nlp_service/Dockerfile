# Base image có kích thước nhỏ nhưng chưa có gcc
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Cài các gói cần thiết để build blis/spacy
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gcc libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements và install Python packages
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Tải data NLTK
RUN python3 -m nltk.downloader punkt stopwords

# Expose port
EXPOSE 8001

# Start Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
