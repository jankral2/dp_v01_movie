# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy requirements files
COPY app/requirements.txt .

# Install Python dependencies with uv (much faster than pip)
# Uses BuildKit cache mount to persist downloads between builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --link-mode=copy -r requirements.txt

# Copy application code
COPY app/ .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
