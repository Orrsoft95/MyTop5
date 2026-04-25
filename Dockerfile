FROM python:3.10-slim

WORKDIR /app

# Install gcc c-compiler & build tools for scikit-surprise
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

CMD["/bin/bash", "/app/start.sh"]