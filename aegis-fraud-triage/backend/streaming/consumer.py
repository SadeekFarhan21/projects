"""Kafka consumer wrapper for transaction and alert streams"""

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class TransactionConsumer:
    """Consumes transactions from Kafka topic"""

    def __init__(self, bootstrap_servers: str, topic: str = "transactions", group_id: str = "aegis"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None

    async def start(self):
        from aiokafka import AIOKafkaConsumer
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
        )
        await self.consumer.start()
        logger.info(f"Consumer started on topic: {self.topic}")

    async def stop(self):
        if self.consumer:
            await self.consumer.stop()
        logger.info("Consumer stopped")

    async def __aiter__(self) -> AsyncIterator[dict]:
        async for msg in self.consumer:
            yield msg.value
