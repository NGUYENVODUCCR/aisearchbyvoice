1. Base image
FROM python:3.9-slim

# 2. Thiết lập working directory
WORKDIR /app

# 3. Copy requirements và install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy source code
COPY . /app

# 5. expose port
EXPOSE 8001

# 6. Khởi động Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]