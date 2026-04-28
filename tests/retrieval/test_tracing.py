"""Tests for OTel tracing helpers."""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from retrieval.observability import get_tracer, init_tracing
from retrieval.observability.tracing import (
    SPAN_BM25,
    SPAN_QUERY,
    record_retrieval_span,
    retrieval_span,
)


@pytest.fixture
def in_memory_provider():
    """In-memory OTel provider with a local tracer (does NOT set the global)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    local_tracer = provider.get_tracer("test_tracer")
    yield local_tracer, exporter
    exporter.clear()


def test_init_tracing_returns_provider():
    provider = init_tracing(service_name="test-service", otlp_endpoint="localhost:9999")
    assert isinstance(provider, TracerProvider)


def test_get_tracer_returns_tracer():
    assert get_tracer("test") is not None


def test_retrieval_span_records_duration(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_BM25, query="test query", top_k=10, tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert "span.duration_ms" in attrs
    assert attrs["span.duration_ms"] >= 0.0


def test_retrieval_span_sets_query_text(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_BM25, query="fever symptoms", tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert spans[0].attributes.get("query.text") == "fever symptoms"


def test_retrieval_span_sets_top_k(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_QUERY, top_k=100, tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert spans[0].attributes.get("query.top_k") == 100


def test_retrieval_span_truncates_long_query(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_BM25, query="x" * 500, tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans[0].attributes.get("query.text", "")) <= 200


def test_retrieval_span_sets_extra_attrs(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_BM25, custom_key="custom_value", tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert spans[0].attributes.get("custom_key") == "custom_value"


def test_retrieval_span_name(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    with retrieval_span(SPAN_BM25, tracer=local_tracer):
        pass

    spans = exporter.get_finished_spans()
    assert spans[0].name == SPAN_BM25


def test_record_retrieval_span_sets_duration(in_memory_provider):
    local_tracer, exporter = in_memory_provider
    record_retrieval_span(SPAN_BM25, duration_ms=42.5, result_count=100, tracer=local_tracer)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("span.duration_ms") == 42.5
    assert spans[0].attributes.get("results.count") == 100


def test_span_name_constants_defined():
    from retrieval.observability.tracing import (
        SPAN_ACL,
        SPAN_BM25,
        SPAN_DENSE_ENCODE,
        SPAN_FAISS_SEARCH,
        SPAN_QUERY,
        SPAN_RRF,
    )
    for s in [SPAN_ACL, SPAN_BM25, SPAN_DENSE_ENCODE, SPAN_FAISS_SEARCH, SPAN_QUERY, SPAN_RRF]:
        assert isinstance(s, str) and len(s) > 0
