"""Konfiguracja kolejki zadań RabbitMQ dla rozproszonej klasyfikacji e-commerce."""

import os
from pathlib import Path

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

# Nazwy kolejek
CLASSIFICATION_QUEUE = "erli_product_classification"
DETECTION_QUEUE = "erli_product_detection"
RESULTS_QUEUE = "erli_classification_results"

# Parametry skalowania
DEFAULT_PREFETCH_COUNT = int(os.getenv("WORKER_PREFETCH", "32"))
