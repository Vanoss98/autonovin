# Use an official Python runtime as a parent image
FROM python:3.12.7

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for psycopg2 and Redis
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 for the Django app
EXPOSE 8000

# Command to run when the container starts
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
