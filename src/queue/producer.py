"""Producent zadań klasyfikacji dla RabbitMQ:
Publikuje zadania klasyfikacji produktów (np. z metadata.jsonl lub napływających na bieżąco ofert)
do rozproszonej kolejki RabbitMQ.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

# Zapewnienie kodowania UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pika

from src.common.config import DEFAULT_METADATA_PATH, PROJECT_ROOT
from src.common.schemas import ProductMetadata, iter_metadata_jsonl
from src.queue.config import (
    CLASSIFICATION_QUEUE,
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
logger = logging.getLogger("queue_producer")


class RabbitMQProducer:
    """Klasa producenta wysyłająca zadania klasyfikacji do RabbitMQ."""

    def __init__(
        self,
        host: str = RABBITMQ_HOST,
        port: int = RABBITMQ_PORT,
        user: str = RABBITMQ_USER,
        password: str = RABBITMQ_PASSWORD,
        queue_name: str = CLASSIFICATION_QUEUE,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.queue_name = queue_name
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

    def connect(self) -> bool:
        """Nawiązuje połączenie z brokerem RabbitMQ."""
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=2,
        )
        try:
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # Deklaracja trwałej kolejki (durable=True przetrwa restart brokera)
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            self.channel.confirm_delivery()
            logger.info("Połączono z brokerem RabbitMQ na %s:%d (kolejka: %s)", self.host, self.port, self.queue_name)
            return True
        except Exception as e:
            logger.error("Nie udało się połączyć z RabbitMQ na %s:%d: %s", self.host, self.port, e)
            return False

    def publish_product(self, product: ProductMetadata) -> bool:
        """Publikuje pojedynczy produkt do kolejki zadań."""
        if not self.channel or self.channel.is_closed:
            if not self.connect():
                return False

        message = product.model_dump_json()
        try:
            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=message.encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Trwałe wiadomości (zapis na dysk)
                    content_type="application/json",
                ),
            )
            return True
        except Exception as e:
            logger.error("Błąd podczas publikacji produktu %s: %s", product.product_id, e)
            return False

    def publish_stream(self, metadata_path: Path, limit: Optional[int] = None) -> int:
        """Publikuje strumień produktów z pliku metadata.jsonl do RabbitMQ."""
        if not self.connect():
            raise ConnectionError(
                f"Brak połączenia z brokerem RabbitMQ na {self.host}:{self.port}. "
                "Upewnij się, że RabbitMQ działa (np. przez Docker: `docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management`)."
            )

        sent_count = 0
        logger.info("Rozpoczynanie strumieniowego zasilania kolejki z %s...", metadata_path)

        for record in iter_metadata_jsonl(metadata_path):
            img_p = PROJECT_ROOT / record.local_image_path
            if not img_p.exists():
                continue

            success = self.publish_product(record)
            if success:
                sent_count += 1
                if sent_count % 100 == 0:
                    print(f"  Wysłano do kolejki RabbitMQ: {sent_count} zadań...", end="\r")

            if limit and sent_count >= limit:
                break

        print(f"\nZakończono publikację. Łącznie wysłano {sent_count} zadań do kolejki {self.queue_name}.")
        return sent_count

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()


def main():
    parser = argparse.ArgumentParser(description="Producent zadań RabbitMQ dla platformy Erli")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Ścieżka do metadata.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Maksymalna liczba zadań do wysłania")
    parser.add_argument("--host", type=str, default=RABBITMQ_HOST, help="Adres hosta RabbitMQ")
    parser.add_argument("--port", type=int, default=RABBITMQ_PORT, help="Port RabbitMQ")
    args = parser.parse_args()

    producer = RabbitMQProducer(host=args.host, port=args.port)
    try:
        producer.publish_stream(args.metadata, limit=args.limit)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
