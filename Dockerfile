# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install filelock requests apscheduler
RUN apt-get update && apt-get install -y tzdata
ENV TZ=UTC
# Copy application code
COPY . .

# Install DVC
RUN pip install --no-cache-dir dvc

# Expose API and Prometheus metrics ports
EXPOSE 5001
EXPOSE 8001

# Set entry point
CMD ["python", "app.py"]
