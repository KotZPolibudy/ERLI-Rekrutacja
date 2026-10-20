"""Worker kolejkowy RabbitMQ dla detekcji elementów ubioru na GPU/CPU.
Obsługuje:
- Przetwarzanie wsadowe (Batching z prefetch) na GPU za pomocą YOLO-World
- Niezależną kolejkę erli_product_detection
- Potwierdzenia dostarczenia (ack) po udanej inferencji (at-least-once delivery)
- Horyzontalne skalowanie workerów w klastrze produkcyjnym
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

# Zapewnienie kodowania UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pika
import torch

from src.common.config import PROJECT_ROOT, REPORTS_DIR
from src.common.schemas import FashionDetectionResult, ProductMetadata
from src.detection.detector import FashionDetector
from src.queue.config import (
    DEFAULT_PREFETCH_COUNT,
    DETECTION_QUEUE,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_USER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("detection_queue_worker")


class RabbitMQDetectionWorker:
    """Rozproszony konsument zadań detekcji ubrań zoptymalizowany pod kątem akceleracji GPU."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        output_file: Path = REPORTS_DIR / "detections_from_queue.jsonl",
        host: str = RABBITMQ_HOST,
        port: int = RABBITMQ_PORT,
        user: str = RABBITMQ_USER,
        password: str = RABBITMQ_PASSWORD,
        queue_name: str = DETECTION_QUEUE,
        conf_threshold: float = 0.25,
    ):
        self.device = device
        self.batch_size = batch_size
        self.output_file = output_file
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.queue_name = queue_name
        self.conf_threshold = conf_threshold

        logger.info("Inicjalizacja workera detekcji mody na urządzeniu: %s (batch_size=%d)...", self.device, self.batch_size)
        self.detector = FashionDetector(
            device=self.device,
            conf_threshold=self.conf_threshold,
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.adapters.blocking_connection.BlockingChannel | None = None

    def connect(self) -> None:
        """Nawiązuje połączenie z brokerem RabbitMQ i deklaruje kolejkę detekcji."""
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300,
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Kolejka trwała (durable=True)
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        # Prefetch count dla efektywnego batchingu na GPU
        self.channel.basic_qos(prefetch_count=max(self.batch_size * 2, DEFAULT_PREFETCH_COUNT))
        logger.info("Połączono z brokerem RabbitMQ na %s:%d. Nasłuchiwanie na kolejce '%s'...", self.host, self.port, self.queue_name)

    def process_batch(self, batch_messages: List[tuple]) -> None:
        """Wykonuje inferencję detektora modowego na paczce obrazów z kolejki."""
        if not batch_messages:
            return

        images_to_process = []
        product_ids = []
        valid_items = []

        for method_frame, properties, body in batch_messages:
            try:
                data = json.loads(body.decode("utf-8"))
                product_id = data.get("product_id", "unknown")
                rel_path = data.get("local_image_path", "")
                full_path = PROJECT_ROOT / rel_path

                if full_path.exists():
                    images_to_process.append(full_path)
                    product_ids.append(product_id)
                    valid_items.append((method_frame, product_id, full_path))
                else:
                    logger.warning("Plik obrazu nie istnieje: %s (id: %s)", full_path, product_id)
                    self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            except Exception as e:
                logger.error("Błąd parsowania wiadomości JSON: %s", e)
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        if not images_to_process:
            return

        t0 = time.perf_counter()
        results: List[FashionDetectionResult] = self.detector.detect_batch(
            images=images_to_process,
            product_ids=product_ids,
            batch_size=self.batch_size,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Zapis do pliku JSONL
        with open(self.output_file, "a", encoding="utf-8") as f:
            for res in results:
                f.write(res.model_dump_json() + "\n")

        # Potwierdzenie przetworzenia wiadomości w RabbitMQ (ACK)
        for method_frame, _, _ in valid_items:
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        total_items = sum(r.item_count for r in results)
        logger.info(
            "Przetworzono paczkę %d obrazów modowych w %.1f ms (%.1f FPS, wykryto %d ubrań).",
            len(images_to_process),
            elapsed_ms,
            len(images_to_process) / max(0.001, elapsed_ms / 1000.0),
            total_items,
        )

    def start_consuming(self) -> None:
        """Pętla pobierania i batchowania zadań detekcji z kolejki RabbitMQ."""
        if not self.channel:
            self.connect()

        logger.info("Worker detekcji gotowy do pracy. Oczekiwanie na zadania w kolejce '%s'...", self.queue_name)

        batch = []
        last_flush_time = time.time()

        try:
            while True:
                method_frame, properties, body = self.channel.basic_get(queue=self.queue_name, auto_ack=False)

                if method_frame:
                    batch.append((method_frame, properties, body))
                    if len(batch) >= self.batch_size:
                        self.process_batch(batch)
                        batch = []
                        last_flush_time = time.time()
                else:
                    if batch and (time.time() - last_flush_time > 1.0):
                        self.process_batch(batch)
                        batch = []
                        last_flush_time = time.time()
                    time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("Zatrzymywanie workera detekcji na żądanie użytkownika...")
            if batch:
                self.process_batch(batch)
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            logger.info("Połączenie z RabbitMQ zostało bezpiecznie zamknięte.")


def main():
    parser = argparse.ArgumentParser(description="Worker kolejkowy RabbitMQ dla detekcji elementów ubioru")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar paczki dla inferencji na GPU")
    parser.add_argument("--queue", type=str, default=DETECTION_QUEUE, help="Nazwa kolejki w RabbitMQ")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "detections_from_queue.jsonl")
    args = parser.parse_args()

    worker = RabbitMQDetectionWorker(
        device=args.device,
        batch_size=args.batch_size,
        queue_name=args.queue,
        output_file=args.output,
    )
    worker.start_consuming()


if __name__ == "__main__":
    main()
