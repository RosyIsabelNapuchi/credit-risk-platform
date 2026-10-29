# Imagen de la API de scoring. Construir:  docker build -t credit-risk-api .
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Entrena el modelo dentro de la imagen si el artefacto no viene en el repo
RUN test -f artifacts/pd_model.joblib || python train.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
