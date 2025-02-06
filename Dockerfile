# Use stable Python image
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip and install the correct NumPy version
RUN pip install --no-cache-dir --upgrade pip \
    && pip uninstall -y numpy \
    && pip install --no-cache-dir numpy==1.26.2 \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
