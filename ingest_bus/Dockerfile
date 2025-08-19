FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/jobs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data/jobs
ENV METRICS_PORT=8081

# Expose ports
EXPOSE 8000
EXPOSE 8081

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.main"]
