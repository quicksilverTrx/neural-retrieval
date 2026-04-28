"""OpenTelemetry distributed tracing for the retrieval pipeline.

Stage boundaries — every stage with latency >1ms gets its own span:
    1. bm25_retrieval — inverted index lookup + BM25 scoring
    2. dense_encode   — query embedding with sentence encoder
    3. faiss_search   — FAISS IVF-PQ approximate nearest-neighbour
    4. rrf_fusion     — RRF rank fusion of BM25 and dense results
    5. acl_filter     — post-retrieval ACL permission filtering
    6. full_query     — root span covering the entire query path

See `docs/design_decisions.md` for the rationale on stage granularity.

Usage
-----
    from retrieval.observability import init_tracing, get_tracer
    from retrieval.observability.tracing import retrieval_span

    init_tracing(service_name="neural-retrieval", otlp_endpoint="localhost:4317")

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("bm25_retrieval") as span:
        span.set_attribute("query.tokens", len(tokens))
        results = ...

Or use the context-manager helpers:

    with retrieval_span("bm25_retrieval", query=query, top_k=top_k) as span:
        results = retriever.retrieve(query, top_k)
        span.set_attribute("results.count", len(results))

Export
------
Spans are exported to a local Jaeger/OTel Collector via OTLP gRPC.
Set OTEL_EXPORTER_OTLP_ENDPOINT (default: grpc://localhost:4317).
If the collector is unreachable the SimpleSpanProcessor swallows errors
silently so the retrieval pipeline is never blocked by telemetry.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Tracer

# Span name constants (prevents typos across the codebase)
SPAN_BM25 = "bm25_retrieval"
SPAN_DENSE_ENCODE = "dense_encode"
SPAN_FAISS_SEARCH = "faiss_search"
SPAN_RRF = "rrf_fusion"
SPAN_ACL = "acl_filter"
SPAN_QUERY = "full_query"

_TRACER_NAME = "neural_retrieval"
_tracer: Tracer | None = None


def init_tracing(
    service_name: str = "neural-retrieval",
    otlp_endpoint: str | None = None,
) -> TracerProvider:
    """Initialise the OTel tracer provider.

    Args:
        service_name:  Reported service name in Jaeger/Tempo.
        otlp_endpoint: gRPC endpoint (e.g. 'localhost:4317').
                       Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var,
                       then 'localhost:4317'.

    Returns:
        The configured TracerProvider (also set as the global provider).
    """
    global _tracer

    endpoint = (
        otlp_endpoint
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
    )

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Attempt OTLP export; fall back to no-op on import failure
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        print(f"[tracing] OTel OTLP exporter → {endpoint}")
    except Exception as exc:
        print(f"[tracing] OTLP exporter unavailable ({exc}). Spans will not be exported.")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_TRACER_NAME)
    return provider


def get_tracer(name: str = _TRACER_NAME) -> Tracer:
    """Return the OTel tracer. Falls back to no-op tracer if not yet configured."""
    if _tracer is None:
        return trace.get_tracer(name)
    return _tracer


@contextmanager
def retrieval_span(
    span_name: str,
    query: str | None = None,
    top_k: int | None = None,
    tracer: Tracer | None = None,
    **extra_attrs,
) -> Generator[trace.Span, None, None]:
    """Context manager that creates a named span with standard retrieval attributes.

    Records:
        query.text         (if provided, truncated to 200 chars)
        query.top_k        (if provided)
        span.duration_ms   (set on exit)
        Any extra kwargs → span attributes

    Args:
        span_name:    OTel span name (use SPAN_* constants from this module).
        query:        Query text to record (truncated to 200 chars).
        top_k:        Top-k value to record.
        tracer:       Optional tracer. Pass a test-local tracer to avoid OTel
                      global provider conflicts.
        **extra_attrs: Additional key-value span attributes.

    Example:
        with retrieval_span(SPAN_BM25, query=query, top_k=100) as span:
            results = bm25_retriever.retrieve(query, 100)
            span.set_attribute("results.count", len(results))
    """
    active_tracer = tracer if tracer is not None else get_tracer()
    t0 = time.perf_counter()

    with active_tracer.start_as_current_span(span_name) as span:
        if query is not None:
            span.set_attribute("query.text", query[:200])
        if top_k is not None:
            span.set_attribute("query.top_k", top_k)
        for key, val in extra_attrs.items():
            span.set_attribute(key, val)

        try:
            yield span
        finally:
            span.set_attribute("span.duration_ms", round((time.perf_counter() - t0) * 1000, 2))


def record_retrieval_span(
    span_name: str,
    duration_ms: float,
    result_count: int = 0,
    query: str | None = None,
    tracer: Tracer | None = None,
) -> None:
    """Fire-and-forget span recorder for callers that already measured latency.

    Useful for code that measures latency internally (e.g. BM25Retriever.retrieve_timed).
    Creates a span that is immediately ended with the provided duration.

    Args:
        tracer: Optional tracer. Pass a test-local tracer to avoid global provider conflicts.
    """
    active_tracer = tracer if tracer is not None else get_tracer()
    with active_tracer.start_as_current_span(span_name) as span:
        span.set_attribute("span.duration_ms", round(duration_ms, 2))
        span.set_attribute("results.count", result_count)
        if query:
            span.set_attribute("query.text", query[:200])
