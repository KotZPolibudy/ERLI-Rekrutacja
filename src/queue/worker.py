"""Worker kolejkowy RabbitMQ dla klasyfikacji obrazów na GPU/CPU.
Obsługuje:
- Przetwarzanie wsadowe (Batching z prefetch) na GPU
- Potwierdzenia dostarczenia (ack) dopiero po udanej inferencji (at-least-once delivery)
- Horyzontalne skalowanie (wiele workerów może czytać z tej samej kolejki bez duplikacji pracy)
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

from src.classification.model import CLIPClassifier, SigLIPClassifier
from src.common.config import PROJECT_ROOT, REPORTS_DIR
from src.common.schemas import CategoryPrediction, ProductMetadata
from src.queue.config import (
    CLASSIFICATION_QUEUE,
    DEFAULT_PREFETCH_COUNT,
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
logger = logging.getLogger("queue_worker")


class RabbitMQClassificationWorker:
    """Rozproszony konsument zadań klasyfikacji zoptymalizowany pod kątem akceleracji GPU."""

    def __init__(
        self,
        model_type: str = "siglip",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        output_file: Path = REPORTS_DIR / "predictions_from_queue.jsonl",
        host: str = RABBITMQ_HOST,
        port: int = RABBITMQ_PORT,
        user: str = RABBITMQ_USER,
        password: str = RABBITMQ_PASSWORD,
        queue_name: str = CLASSIFICATION_QUEUE,
    ):
        self.device = device
        self.batch_size = batch_size
        self.output_file = output_file
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.queue_name = queue_name

        logger.info("Inicjalizacja Workera na urządzeniu: %s (batch_size=%d)...", self.device, self.batch_size)
        if model_type.lower() == "clip":
            self.classifier = CLIPClassifier(device=self.device)
        else:
            self.classifier = SigLIPClassifier(device=self.device)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.total_processed = 0

    def start_consuming(self):
        """Rozpoczyna nasłuchiwanie na zadania z RabbitMQ."""
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )

        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)

            # Prefetch: pobiera paczkę zadań dla workera
            channel.basic_qos(prefetch_count=max(self.batch_size * 2, DEFAULT_PREFETCH_COUNT))
            logger.info("Worker gotowy. Oczekiwanie na zadania z kolejki '%s'...", self.queue_name)

            buffer_tags: List[int] = []
            buffer_records: List[ProductMetadata] = []

            def process_buffer():
                if not buffer_records:
                    return

                t0 = time.perf_counter()
                b_imgs = [PROJECT_ROOT / r.local_image_path for r in buffer_records]
                b_ids = [r.product_id for r in buffer_records]
                b_cats = [r.category_main for r in buffer_records]

                preds = self.classifier.predict_batch(
                    images=b_imgs,
                    product_ids=b_ids,
                    true_categories=b_cats,
                )

                # Zapisz do pliku wyników
                with self.output_file.open("a", encoding="utf-8") as out_f:
                    for p in preds:
                        out_f.write(p.model_dump_json() + "\n")
                    out_f.flush()

                # Potwierdź wiadomości w RabbitMQ (ack)
                for tag in buffer_tags:
                    channel.basic_ack(delivery_tag=tag)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.total_processed += len(preds)
                fps = len(preds) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
                logger.info(
                    "Przetworzono batch %d produktów w %.1f ms (%.1f FPS) | Łącznie: %d",
                    len(preds), elapsed_ms, fps, self.total_processed
                )

                buffer_tags.clear()
                buffer_records.clear()

            def on_message(ch, method, properties, body):
                try:
                    data = json.loads(body.decode("utf-8"))
                    record = ProductMetadata.model_validate(data)
                    buffer_records.append(record)
                    buffer_tags.append(method.delivery_tag)

                    if len(buffer_records) >= self.batch_size:
                        process_buffer()

                except Exception as e:
                    logger.error("Błąd przetwarzania wiadomości: %s", e)
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=self.queue_name, on_message_callback=on_message)

            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                logger.info("Zatrzymywanie workera...")
                if buffer_records:
                    process_buffer()
                channel.stop_consuming()
                connection.close()
                logger.info("Worker zatrzymany poprawnie.")

        except Exception as e:
            logger.error(
                "Nie udało się połączyć z brokerem RabbitMQ na %s:%d: %s\n"
                "Podpowiedź: Uruchom kontener RabbitMQ za pomocą Docker:\n"
                "    docker run -d --name erli-rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management",
                self.host, self.port, e
            )


def main():
    parser = argparse.ArgumentParser(description="Worker klasyfikacji produktów Erli (RabbitMQ + GPU)")
    parser.add_argument("--model", type=str, default="siglip", choices=["siglip", "clip"], help="Model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Urządzenie (cuda / cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Wielkość paczki dla inferencji")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "predictions_from_queue.jsonl", help="Plik wyników")
    parser.add_argument("--host", type=str, default=RABBITMQ_HOST, help="Adres hosta RabbitMQ")
    parser.add_argument("--port", type=int, default=RABBITMQ_PORT, help="Port RabbitMQ")
    args = parser.parse_args()

    worker = RabbitMQClassificationWorker(
        model_type=args.model,
        device=args.device,
        batch_size=args.batch_size,
        output_file=args.output,
        host=args.host,
        port=args.port,
    )
    worker.start_consuming()


if __name__ == "__main__":
    main()
