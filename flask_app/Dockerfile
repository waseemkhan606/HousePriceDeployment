# Use an official Python image
FROM python:3.9-slim

WORKDIR /app

# Copy only the requirements first
COPY requirements.txt .

# Upgrade pip separately to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies safely
RUN pip install --no-cache-dir -r requirements.txt || echo "Dependency installation failed!"

# Copy the rest of the project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
