FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY start_participant.sh .
COPY start_all.sh .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Make scripts executable
RUN chmod +x start_participant.sh start_all.sh

EXPOSE 8000

ENTRYPOINT ["bash", "start_participant.sh"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
