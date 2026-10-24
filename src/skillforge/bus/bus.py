"""Event bus abstraction: in-memory (tests, experiments) and Kafka API (Redpanda, deployment)."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Protocol

TRAJECTORIES = "trajectories.finished"
CANDIDATES = "skills.candidates"
SKILL_EVENTS = "skills.events"


class Consumer(Protocol):
    def poll(self, max_messages: int = 100, timeout: float = 0.0) -> list[dict]: ...
    def commit(self) -> None: ...


class Bus(Protocol):
    def publish(self, topic: str, value: dict, key: str | None = None) -> None: ...
    def consumer(self, topic: str, group: str) -> Consumer: ...
    def flush(self) -> None: ...


class _MemConsumer:
    def __init__(self, bus: "InMemoryBus", topic: str, group: str):
        self.bus, self.topic, self.group = bus, topic, group

    def poll(self, max_messages: int = 100, timeout: float = 0.0) -> list[dict]:
        off = self.bus.offsets[(self.topic, self.group)]
        msgs = self.bus.topics[self.topic][off:off + max_messages]
        self._taken = len(msgs)
        return [json.loads(m) for m in msgs]

    def commit(self) -> None:
        self.bus.offsets[(self.topic, self.group)] += getattr(self, "_taken", 0)
        self._taken = 0


class InMemoryBus:
    """Ordered, replayable, at-least-once. Values are JSON round-tripped so it behaves like a real broker."""

    def __init__(self):
        self.topics: dict[str, list[str]] = defaultdict(list)
        self.offsets: dict[tuple[str, str], int] = defaultdict(int)

    def publish(self, topic, value, key=None):
        self.topics[topic].append(json.dumps(value))

    def consumer(self, topic, group):
        return _MemConsumer(self, topic, group)

    def flush(self):
        pass


class _KafkaConsumer:
    def __init__(self, c):
        self.c = c

    def poll(self, max_messages: int = 100, timeout: float = 1.0) -> list[dict]:
        out = []
        for m in self.c.consume(num_messages=max_messages, timeout=timeout):
            if m.error():
                continue
            out.append(json.loads(m.value()))
        return out

    def commit(self) -> None:
        self.c.commit(asynchronous=False)


class KafkaBus:
    """Speaks the Kafka protocol; works against Redpanda or Apache Kafka."""

    def __init__(self, brokers: str | None = None):
        from confluent_kafka import Producer

        self.brokers = brokers or os.environ.get("KAFKA_BROKERS", "localhost:19092")
        self.producer = Producer({"bootstrap.servers": self.brokers, "linger.ms": 5})

    def publish(self, topic, value, key=None):
        self.producer.produce(topic, json.dumps(value).encode(), key=key.encode() if key else None)
        self.producer.poll(0)

    def consumer(self, topic, group):
        from confluent_kafka import Consumer

        c = Consumer({"bootstrap.servers": self.brokers, "group.id": group, "auto.offset.reset": "earliest",
                      "enable.auto.commit": False})
        c.subscribe([topic])
        return _KafkaConsumer(c)

    def flush(self):
        self.producer.flush(10)


def wait_for_kafka(brokers: str, seconds: float = 60.0) -> None:
    from confluent_kafka.admin import AdminClient

    end = time.time() + seconds
    while time.time() < end:
        try:
            AdminClient({"bootstrap.servers": brokers}).list_topics(timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Kafka/Redpanda not reachable at {brokers}")
