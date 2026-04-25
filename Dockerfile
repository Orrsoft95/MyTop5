FROM python:3.10-slim

WORKDIR /app

# Install gcc c-compiler & build tools for scikit-surprise
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create .streamlit/secrets.toml within Docker Container & write secrets from env variables
RUN mkdir -p /app/.streamlit

CMD mkdir -p /app/.streamlit && echo "[mal]" > /app/.streamlit/secrets.toml && \
    echo "client_id = \"${mal__client_id}\"" >> /app/.streamlit/secrets.toml && \
    echo "[huggingface]" >> /app/.streamlit/secrets.toml && \
    echo "repo_id = \"${huggingface__repo_id}\"" >> /app/.streamlit/secrets.toml && \ 

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]