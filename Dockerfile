FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt redis
COPY . .
RUN useradd -m quant && chown -R quant /app
USER quant
EXPOSE 9108
ENTRYPOINT ["python", "main.py"]
CMD ["--paper", "--config", "config/strategy_etf.yaml", "--broker", "sim", "--metrics-port", "9108"]
