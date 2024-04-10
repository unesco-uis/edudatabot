# app/uisedubot
# Use an official Python runtime as a parent image 
FROM python:3.10-slim

# Set environment variables 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch main https://github.com/unesco-uis/edudatabot.git

# Set the working directory in the container 
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .
COPY main.py .

# Install requirements from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port that the application will listen on 
EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--theme.base=light"]
