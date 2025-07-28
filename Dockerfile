FROM python:3.11-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd -r -s /bin/false -u 1000 ocruser

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements_optimized.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_optimized.txt

# Copiar código da aplicação
COPY ocr_service.py .

# Definir permissões
RUN chmod +x ocr_service.py \
    && chown -R ocruser:ocruser /app

# Variáveis de ambiente para performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MAX_WORKERS=8
ENV CACHE_TTL=3600

# Expor porta
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Usuário não-root
USER ocruser

# Comando de inicialização
CMD ["python", "ocr_service.py"] 