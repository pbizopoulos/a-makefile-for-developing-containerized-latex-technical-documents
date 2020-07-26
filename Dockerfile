FROM python:3
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
