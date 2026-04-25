#!/bin/bash
mkdir -p /app/.streamlit
cat > /app/.streamlit/secrets.toml << EOF
[mal]
client_id = "${mal__client_id}"

[huggingface]
repo_id = "${huggingface__repo_id}"
EOF
streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false