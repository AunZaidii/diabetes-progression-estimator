# Container for the Hugging Face Space (also works locally: docker build -t diabetes-app . && docker run -p 8501:8501 diabetes-app)
FROM python:3.14-slim

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Train inside the container so the saved model matches its scikit-learn version
RUN python train.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
