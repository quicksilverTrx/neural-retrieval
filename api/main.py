"""FastAPI serving layer for the neural retrieval engine.

Endpoints:
    POST /search    Hybrid BM25+dense retrieval with optional ACL filter
    GET  /health    Liveness probe (process alive)
    GET  /ready     Readiness probe (indexes loaded)
    GET  /metrics   Prometheus-style text metrics

Startup:
    uvicorn api.main:app --host 0.0.0.0 --port 8080

Environment variables (all optional):
    BM25_INDEX_PATH    Path to custom_bm25_8m.bin
    FAISS_INDEX_DIR    Path to FAISS index directory
    ENCODER_MODEL      HF model name (default: all-MiniLM-L6-v2)
    ACL_DATA_DIR       Path to ACL data directory (ACL disabled if absent)
    DEFAULT_TOP_K      Default number of results (default: 10)
    DEFAULT_NPROBE     FAISS nprobe (default: 16)
    RRF_K              RRF k parameter (default: 60)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Structured JSON logging — one object per record
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_setup_logging()
_log = logging.getLogger("neural-retrieval.api")

from retrieval.observability.tracing import (
    SPAN_ACL,
    SPAN_BM25,
    SPAN_DENSE_ENCODE,
    SPAN_FAISS_SEARCH,
    SPAN_QUERY,
    SPAN_RRF,
    init_tracing,
    retrieval_span,
)


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

_BM25_PATH = Path(os.getenv(
    "BM25_INDEX_PATH", str(REPO_ROOT / "data" / "custom_bm25_8m.bin")
))
_ENCODER_MODEL = os.getenv("ENCODER_MODEL", "all-MiniLM-L6-v2")
_FAISS_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(
    REPO_ROOT / "data" / "faiss" /
    _ENCODER_MODEL.replace("/", "_").replace("-", "_").lower()
)))
_ACL_DIR = Path(os.getenv("ACL_DATA_DIR", str(REPO_ROOT / "data" / "acl")))
_DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
_DEFAULT_NPROBE = int(os.getenv("DEFAULT_NPROBE", "16"))
_RRF_K = int(os.getenv("RRF_K", "60"))


# ---------------------------------------------------------------------------
# Module-level state — loaded once at startup, read by every request handler
# ---------------------------------------------------------------------------

class _State:
    bm25_retriever = None
    dense_encoder = None
    faiss_index = None
    acl_filter = None
    ready: bool = False
    startup_error: str | None = None

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0


_state = _State()


# ---------------------------------------------------------------------------
# Lifespan — load indexes once on startup; ship corruption-recovery here
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys
    sys.path.insert(0, str(REPO_ROOT))

    init_tracing(service_name="neural-retrieval")

    # ---- BM25 ----
    try:
        from retrieval.inverted_index import BM25Retriever, BM25Scorer
        from retrieval.inverted_index.persistence import load_index

        if _BM25_PATH.exists():
            _log.info(json.dumps({"event": "startup.bm25.loading", "path": str(_BM25_PATH)}))
            index, _, _ = load_index(_BM25_PATH)
            _state.bm25_retriever = BM25Retriever(index=index, scorer=BM25Scorer())
            _log.info(json.dumps({"event": "startup.bm25.loaded", "num_docs": index.num_docs}))
        else:
            _log.warning(json.dumps({"event": "startup.bm25.missing", "path": str(_BM25_PATH)}))
    except Exception as e:
        _state.startup_error = f"BM25 load failed: {e}"
        _log.error(json.dumps({"event": "startup.bm25.error", "error": str(e)}))

    # ---- Dense (encoder + FAISS, with corruption recovery) ----
    try:
        from retrieval.dense.encoder import SentenceEncoder
        from retrieval.dense.faiss_index import FAISSIVFPQIndex

        _log.info(json.dumps({"event": "startup.encoder.loading", "model": _ENCODER_MODEL}))
        _state.dense_encoder = SentenceEncoder(model_name=_ENCODER_MODEL)

        if _FAISS_DIR.exists():
            _log.info(json.dumps({"event": "startup.faiss.loading", "path": str(_FAISS_DIR)}))
            from retrieval.dense.recovery import rebuild_index, validate_faiss_index
            try:
                validate_faiss_index(_FAISS_DIR)
            except Exception as val_err:
                _log.warning(json.dumps({
                    "event": "startup.faiss.checksum_mismatch",
                    "error": str(val_err),
                }))
                rebuild_index(_FAISS_DIR, encoder=_state.dense_encoder)
            _state.faiss_index = FAISSIVFPQIndex.load(_FAISS_DIR)
            _log.info(json.dumps({
                "event": "startup.faiss.loaded",
                "ntotal": _state.faiss_index._index.ntotal,
            }))
        else:
            _log.warning(json.dumps({"event": "startup.faiss.missing", "path": str(_FAISS_DIR)}))
    except Exception as e:
        _state.startup_error = f"Dense load failed: {e}"
        _log.error(json.dumps({"event": "startup.dense.error", "error": str(e)}))

    # ---- ACL (optional — server runs fine without it) ----
    try:
        from retrieval.acl import ACLFilter, PassageACL

        if (_ACL_DIR / "passage_acl.json").exists():
            _log.info(json.dumps({"event": "startup.acl.loading", "path": str(_ACL_DIR)}))
            acl = PassageACL()
            acl.load(_ACL_DIR)
            _state.acl_filter = ACLFilter(acl)
            _log.info(json.dumps({
                "event": "startup.acl.loaded", "num_passages": acl.num_passages,
            }))
        else:
            _log.info(json.dumps({"event": "startup.acl.disabled", "reason": "no passage_acl.json"}))
    except Exception as e:
        _log.warning(json.dumps({"event": "startup.acl.error", "error": str(e)}))

    _state.ready = _state.startup_error is None
    _log.info(json.dumps({"event": "startup.complete", "ready": _state.ready}))

    yield

    _log.info(json.dumps({"event": "shutdown"}))


app = FastAPI(
    title="Neural Retrieval API",
    description="Custom BM25 + FAISS IVF-PQ + hybrid fusion",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Correlation-ID middleware — extracts/mints X-Request-ID, propagates to OTel
# ---------------------------------------------------------------------------

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        from opentelemetry import trace

        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("http.request_id", correlation_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response


app.add_middleware(CorrelationIDMiddleware)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048,
                       description="Search query text")
    top_k: int = Field(default=_DEFAULT_TOP_K, ge=1, le=1000)
    user_role: Optional[str] = Field(default=None,
                                     description="Requesting user's role (for ACL)")
    mode: str = Field(default="hybrid",
                      description="Retrieval mode: 'bm25', 'dense', or 'hybrid' (RRF)")
    nprobe: int = Field(default=_DEFAULT_NPROBE, ge=1, le=512,
                        description="FAISS nprobe (dense/hybrid only)")


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    mode: str
    latency_ms: float
    num_results: int


# ---------------------------------------------------------------------------
# Per-mode retrieval helpers — each opens its own span
# ---------------------------------------------------------------------------

def _bm25_retrieve(query: str, top_k: int) -> list[tuple[str, float]]:
    if _state.bm25_retriever is None:
        raise HTTPException(status_code=503, detail="BM25 index not loaded")
    with retrieval_span(SPAN_BM25, query=query, top_k=top_k) as span:
        results, latency_ms = _state.bm25_retriever.retrieve_timed(query, top_k=top_k)
        span.set_attribute("results.count", len(results))
        span.set_attribute("bm25.latency_ms", round(latency_ms, 2))
        # BM25Retriever returns int doc_ids; FAISS pids.json stores str. Normalise.
        return [(str(d), s) for d, s in results]


def _dense_retrieve(query: str, top_k: int, nprobe: int) -> list[tuple[str, float]]:
    if _state.dense_encoder is None or _state.faiss_index is None:
        raise HTTPException(status_code=503, detail="Dense index not loaded")
    with retrieval_span(SPAN_DENSE_ENCODE, query=query) as span:
        vec = _state.dense_encoder.encode_query(query)
        span.set_attribute("embedding.dim", vec.shape[-1])
    with retrieval_span(SPAN_FAISS_SEARCH, top_k=top_k, nprobe=nprobe) as span:
        pids_per_q, dists = _state.faiss_index.search(vec, top_k=top_k, nprobe=nprobe)
        span.set_attribute("results.count", len(pids_per_q[0]))
    return list(zip(pids_per_q[0], dists[0].tolist()))


def _hybrid_retrieve(query: str, top_k: int, nprobe: int) -> list[tuple[str, float]]:
    """RRF hybrid: retrieve top_k*2 from each leg, fuse, return top_k."""
    from retrieval.fusion.rrf import fuse_scored

    oversample = top_k * 2
    bm25_results = _bm25_retrieve(query, top_k=oversample)
    dense_results = _dense_retrieve(query, top_k=oversample, nprobe=nprobe)

    with retrieval_span(SPAN_RRF, top_k=top_k) as span:
        fused = fuse_scored(bm25_results, dense_results, k=_RRF_K)
        span.set_attribute("rrf.k", _RRF_K)
        span.set_attribute("results.count", len(fused))

    return fused[:top_k]


def _apply_acl(
    results: list[tuple[str, float]],
    user_role: str | None,
    top_k: int,
) -> list[tuple[str, float]]:
    with retrieval_span(SPAN_ACL, top_k=top_k) as span:
        acl_enabled = user_role is not None and _state.acl_filter is not None
        span.set_attribute("acl.enabled", acl_enabled)
        if not acl_enabled:
            filtered = results[:top_k]
        else:
            filtered = _state.acl_filter.filter(results, user_role=user_role, top_k=top_k)
        span.set_attribute("results.count", len(filtered))
        return filtered


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Retrieve passages for a query.

    Modes:
        - bm25:   Custom inverted index BM25 (exact match, fast)
        - dense:  FAISS IVF-PQ semantic search
        - hybrid: RRF fusion of BM25 + dense
    """
    _state.request_count += 1
    t0 = time.perf_counter()

    try:
        with retrieval_span(
            SPAN_QUERY,
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
        ) as root_span:
            oversample = request.top_k * 2  # extra to survive ACL filter
            if request.mode == "bm25":
                raw = _bm25_retrieve(request.query, top_k=oversample)
            elif request.mode == "dense":
                raw = _dense_retrieve(request.query, top_k=oversample, nprobe=request.nprobe)
            elif request.mode == "hybrid":
                raw = _hybrid_retrieve(request.query, top_k=oversample, nprobe=request.nprobe)
            else:
                raise HTTPException(
                    status_code=422, detail=f"Unknown mode '{request.mode}'"
                )

            filtered = _apply_acl(raw, request.user_role, top_k=request.top_k)
            root_span.set_attribute("response.count", len(filtered))

    except HTTPException:
        _state.error_count += 1
        raise
    except Exception as exc:
        _state.error_count += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _state.total_latency_ms += elapsed_ms

    results = [
        SearchResult(doc_id=doc_id, score=float(score), rank=i + 1)
        for i, (doc_id, score) in enumerate(filtered)
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        mode=request.mode,
        latency_ms=round(elapsed_ms, 2),
        num_results=len(results),
    )


@app.get("/health")
async def health():
    """Liveness probe — 200 if process is alive."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe — 200 when indexes are loaded, 503 otherwise."""
    if not _state.ready:
        raise HTTPException(
            status_code=503,
            detail=_state.startup_error or "Indexes not yet loaded",
        )
    loaded = {
        "bm25": _state.bm25_retriever is not None,
        "dense": _state.faiss_index is not None,
        "acl": _state.acl_filter is not None,
    }
    return {"status": "ready", "loaded": loaded}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus-style text metrics."""
    mean_lat = (
        _state.total_latency_ms / _state.request_count
        if _state.request_count > 0 else 0.0
    )
    lines = [
        "# HELP retrieval_requests_total Total search requests",
        "# TYPE retrieval_requests_total counter",
        f"retrieval_requests_total {_state.request_count}",
        "",
        "# HELP retrieval_errors_total Total search errors",
        "# TYPE retrieval_errors_total counter",
        f"retrieval_errors_total {_state.error_count}",
        "",
        "# HELP retrieval_latency_ms_mean Mean search latency in ms",
        "# TYPE retrieval_latency_ms_mean gauge",
        f"retrieval_latency_ms_mean {mean_lat:.2f}",
        "",
        "# HELP retrieval_index_ready Index readiness flag (1=ready)",
        "# TYPE retrieval_index_ready gauge",
        f"retrieval_index_ready {1 if _state.ready else 0}",
    ]
    return "\n".join(lines)
