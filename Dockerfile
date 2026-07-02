FROM python:3.12-slim

# tesseract-ocr: OCR fallback in doc_extract.py.
# iputils-ping, dnsutils, whois: binaries network_tool() shells out to (ping/dig/whois).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    iputils-ping \
    dnsutils \
    whois \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
