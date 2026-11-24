FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY examples ./examples
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["echobench", "serve", "--host", "0.0.0.0", "--port", "8000"]
