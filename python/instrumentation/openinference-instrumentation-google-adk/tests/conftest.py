from typing import Iterator

import pytest
from _pytest.monkeypatch import MonkeyPatch
from opentelemetry import metrics as metrics_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk import metrics as metrics_sdk
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.google_adk import GoogleADKInstrumentor


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def in_memory_metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture
def meter_provider(
    in_memory_metric_reader: InMemoryMetricReader,
) -> metrics_api.MeterProvider:
    meter_provider = metrics_sdk.MeterProvider(metric_readers=[in_memory_metric_reader])
    return meter_provider


@pytest.fixture
def instrument(
    tracer_provider: trace_api.TracerProvider,
    meter_provider: metrics_api.MeterProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    in_memory_metric_reader: InMemoryMetricReader,
) -> Iterator[None]:
    GoogleADKInstrumentor().instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
    yield
    GoogleADKInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "xyz")
