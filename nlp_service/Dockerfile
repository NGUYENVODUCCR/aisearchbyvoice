# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Expose port
EXPOSE 8001

# Start Uvicorn server
CMD ["uvicorn", "nlp_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
