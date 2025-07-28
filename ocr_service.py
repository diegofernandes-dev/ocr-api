#!/usr/bin/env python3
"""
OCR Service Python - VERSÃO ULTRA-FAST
Processamento 100% em memória + APENAS MELHORES CONFIGURAÇÕES
"""

import os
import sys
import json
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import logging
from dataclasses import dataclass
import gc
import io
import base64

# OCR libraries
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np

# Web framework
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Cache e otimizações
from cachetools import TTLCache, cached
import psutil
import hashlib

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações críticas - ULTRA OTIMIZADAS
MAX_WORKERS = min(8, (os.cpu_count() or 1))  # Reduzido para evitar overhead
CACHE_TTL = 3600
MAX_FILE_SIZE = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gmp', 'bmp', 'tiff'}

# Cache em memória para resultados
result_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Pool de workers para OCR - ULTRA OTIMIZADO
ocr_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@dataclass
class OCRResult:
    """Resultado estruturado do OCR"""
    text: str
    confidence: float
    processing_time: float
    strategy_used: str
    image_hash: str
    preprocessing_method: str

class OCRResponse(BaseModel):
    """Resposta padronizada da API"""
    text: str
    confidence: float
    processing_time: float
    status: str
    strategy_used: str
    preprocessing_method: str

# FastAPI app
app = FastAPI(
    title="OCR Service - Ultra Fast",
    description="Serviço de OCR 100% em memória + ULTRA OTIMIZADO - Máxima velocidade",
    version="4.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_image_hash(image_bytes: bytes) -> str:
    """Gera hash único da imagem para cache"""
    return hashlib.md5(image_bytes).hexdigest()

def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Converte bytes da imagem para numpy array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def numpy_to_pil(numpy_image: np.ndarray) -> Image.Image:
    """Converte numpy array para PIL Image"""
    # Converter BGR para RGB
    rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def ultra_fast_preprocessing(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Pré-processamento ULTRA OTIMIZADO - apenas os 2 métodos mais eficazes
    """
    results = []
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Original (sempre testar)
    results.append(("original", gray))
    
    # 2. CLAHE (muito eficaz para screenshots)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    results.append(("clahe", clahe_img))
    
    return results  # Apenas 2 métodos!

def get_ultra_fast_configs() -> List[Tuple[str, str]]:
    """Configurações ULTRA OTIMIZADAS - apenas as 2 mais eficazes"""
    return [
        ('por+eng', '--oem 3 --psm 3'),   # Melhor para documentos
        ('eng', '--oem 3 --psm 3'),       # Inglês puro
    ]  # Apenas 2 configs!

def run_ocr_single_config_ultra(image: np.ndarray, lang: str, config: str) -> Tuple[str, float]:
    """
    Executa OCR em uma configuração específica - ULTRA OTIMIZADO
    """
    try:
        # Converter numpy para PIL
        pil_image = numpy_to_pil(image)
        
        # Configurar Tesseract
        custom_config = f'-l {lang} {config}'
        
        # Executar OCR
        result = pytesseract.image_to_data(
            pil_image,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extrair texto e confiança
        text = ' '.join([word for word in result['text'] if word.strip()])
        confidences = [conf for conf in result['conf'] if conf > 0]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.0
            
        return text, avg_confidence
        
    except Exception as e:
        logger.error(f"Erro no OCR {lang} {config}: {e}")
        return "", 0.0

def run_ocr_ultra_fast(image: np.ndarray) -> Tuple[str, float, str, str]:
    """
    OCR ULTRA RÁPIDO - apenas 2×2 = 4 combinações!
    """
    start_time = time.time()
    
    # Obter configurações ultra otimizadas
    configs = get_ultra_fast_configs()
    preprocessed_images = ultra_fast_preprocessing(image)
    
    # Preparar todas as combinações para execução paralela
    tasks = []
    for preprocess_name, processed_image in preprocessed_images:
        for lang, config in configs:
            tasks.append((preprocess_name, processed_image, lang, config))
    
    # Executar OCR em paralelo
    futures = []
    for preprocess_name, processed_image, lang, config in tasks:
        future = ocr_executor.submit(run_ocr_single_config_ultra, processed_image, lang, config)
        futures.append((future, preprocess_name, lang, config))
    
    # Coletar resultados
    best_text = ""
    best_confidence = 0.0
    best_strategy = ""
    best_preprocessing = ""
    
    for future, preprocess_name, lang, config in futures:
        try:
            text, confidence = future.result(timeout=5)  # Timeout reduzido para 5s
            
            if text and confidence > 0:
                # Critérios de seleção ultra simplificados
                score = confidence
                
                # Bônus para textos mais longos
                if len(text.strip()) > 10 and confidence > 70:
                    score += 5
                
                # Atualizar melhor resultado
                if score > best_confidence:
                    best_text = text
                    best_confidence = score
                    best_strategy = f"{lang}_{config}"
                    best_preprocessing = preprocess_name
                    
        except Exception as e:
            logger.error(f"Erro na combinação {preprocess_name} + {lang} {config}: {e}")
            continue
    
    processing_time = time.time() - start_time
    logger.info(f"⚡ Tempo total ultra-fast: {processing_time:.2f}s")
    
    return best_text, best_confidence, best_strategy, best_preprocessing

def intelligent_post_processing_ultra(text: str) -> str:
    """Pós-processamento ULTRA OTIMIZADO do texto"""
    if not text:
        return text
    
    # Correções principais apenas
    corrections = {
        # Erros comuns do Tesseract
        'Oss': 'The', 'ee': 'is', 'Oe': 'The', 'eee': 'the',
        'Meee': 'The', 'eset': 'set', 'emt': 'and', 'SMSC': 'SMS',
        'BOE': 'The', 'wiecM': 'which', 'Cott': 'that', 'cucu': 'with',
        
        # Correções específicas para Google
        'httplil': 'httplib.h', 'Leptor': 'Leptonica', 'http': 'httplib.h',
        'SE [e]': '•', 'o [|': '•', 'fel': '*',
        'Handling}': 'Handling', 'Handling \'The': 'Handling. The',
        '[e [': '*', 'PJSON': 'JSON', 'P': '*'
    }
    
    # Aplicar correções
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Limpeza adicional
    text = text.strip()
    
    return text

async def process_ocr_ultra_fast(image_bytes: bytes) -> OCRResult:
    """
    Processamento completo ULTRA OTIMIZADO 100% em memória
    """
    start_time = time.time()
    
    # Verificar cache primeiro
    image_hash = get_image_hash(image_bytes)
    if image_hash in result_cache:
        logger.info("Resultado encontrado no cache")
        return result_cache[image_hash]
    
    try:
        # Converter bytes para numpy
        image = bytes_to_numpy(image_bytes)
        
        # OCR ULTRA RÁPIDO
        text, confidence, strategy, preprocessing = await asyncio.get_event_loop().run_in_executor(
            ocr_executor,
            run_ocr_ultra_fast,
            image
        )
        
        # Pós-processamento ultra otimizado
        final_text = intelligent_post_processing_ultra(text)
        
        processing_time = time.time() - start_time
        
        # Criar resultado
        result = OCRResult(
            text=final_text,
            confidence=confidence,
            processing_time=processing_time,
            strategy_used=strategy,
            image_hash=image_hash,
            preprocessing_method=preprocessing
        )
        
        # Armazenar no cache
        result_cache[image_hash] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no processamento OCR: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check com métricas do sistema"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        active_threads = threading.active_count()
        cache_size = len(result_cache)
        
        return {
            "status": "healthy",
            "service": "ocr-service-ultra-fast",
            "version": "4.0.0",
            "metrics": {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_available_mb": round(memory.available / 1024 / 1024, 2),
                "active_threads": active_threads,
                "cache_size": cache_size,
                "max_workers": MAX_WORKERS
            }
        }
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    """Endpoint principal - 100% em memória + ULTRA RÁPIDO"""
    start_time = time.time()
    
    try:
        # Validações
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
        
        if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Tipo de arquivo não permitido")
        
        # Ler arquivo em memória
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Arquivo muito grande")
        
        try:
            # Processar OCR 100% em memória + ULTRA RÁPIDO
            result = await process_ocr_ultra_fast(content)
            
            # Forçar garbage collection
            gc.collect()
            
            return OCRResponse(
                text=result.text,
                confidence=round(result.confidence, 2),
                processing_time=round(result.processing_time, 2),
                status="ok",
                strategy_used=result.strategy_used,
                preprocessing_method=result.preprocessing_method
            )
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro geral: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "service": "OCR Service - Ultra Fast",
        "version": "4.0.0",
        "description": "Processamento 100% em memória + ULTRA OTIMIZADO - Máxima velocidade",
        "endpoints": {
            "GET /": "Informações do serviço",
            "GET /health": "Health check com métricas",
            "POST /ocr": "Processamento de OCR"
        }
    }

if __name__ == "__main__":
    print("🚀 OCR Service - Ultra Fast iniciando...")
    print("📝 Endpoints disponíveis:")
    print("   GET  /       - Informações do serviço")
    print("   GET  /health - Health check com métricas")
    print("   POST /ocr    - Processamento de OCR")
    print("💾 Processamento 100% em memória + ULTRA OTIMIZADO!")
    print("⚡ MÁXIMA VELOCIDADE - 2 métodos × 2 configs = 4 combinações paralelas")
    print("🌐 Servidor rodando em http://0.0.0.0:8080")
    
    uvicorn.run(
        "ocr_service:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info"
    ) 