"""
Passage chunker for neural retrieval.

Splits passages into overlapping token windows using word-boundary tokenization
consistent with the BM25 inverted index (re.findall(r"\\w+", text.lower())).

Design:
- Window size: 256 tokens (word tokens via regex split)
- Stride: 32 tokens (overlap = window - stride = 224 tokens)
- Most MS MARCO v1 passages (~60 tokens avg) pass through as a single chunk.
- Chunk→passage mapping stored so retrieved chunks can be mapped back to
  original passage IDs for deduplication and citation.

The 256/32 parameters are set at construction time to allow ablation
(128/256/512/1024 window sweeps).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str      # "{passage_id}_{chunk_index}"
    passage_id: str
    text: str
    token_start: int   # index into the passage's token list
    token_end: int     # exclusive


def _tokenize(text: str) -> list[str]:
    """Word-boundary tokenization matching the BM25 index tokenizer."""
    return re.findall(r"\w+", text.lower())


class PassageChunker:
    """Chunks passages into overlapping token windows.

    Parameters
    ----------
    window_size : int
        Number of tokens per chunk (default 256).
    stride : int
        Step between chunk starts (default 32). Must be < window_size.
    """

    def __init__(self, window_size: int = 256, stride: int = 32) -> None:
        if stride >= window_size:
            raise ValueError(f"stride ({stride}) must be < window_size ({window_size})")
        self.window_size = window_size
        self.stride = stride

    def chunk_passage(self, passage_id: str, text: str) -> list[ChunkRecord]:
        """Split one passage into overlapping chunks.

        Short passages (len(tokens) <= window_size) return exactly one chunk
        covering the full text — no artificial padding.
        """
        tokens = _tokenize(text)

        if len(tokens) <= self.window_size:
            return [
                ChunkRecord(
                    chunk_id=f"{passage_id}_0",
                    passage_id=passage_id,
                    text=text,
                    token_start=0,
                    token_end=len(tokens),
                )
            ]

        chunks: list[ChunkRecord] = []
        chunk_index = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{passage_id}_{chunk_index}",
                    passage_id=passage_id,
                    text=chunk_text,
                    token_start=start,
                    token_end=end,
                )
            )
            chunk_index += 1
            if end == len(tokens):
                break
            start += self.stride

        return chunks

    def chunk_corpus(
        self,
        passages: list[tuple[str, str]],
    ) -> tuple[list[ChunkRecord], dict[str, str]]:
        """Chunk an entire corpus.

        Parameters
        ----------
        passages : list of (passage_id, text)

        Returns
        -------
        chunks : flat list of all ChunkRecord objects
        chunk_to_passage : dict mapping chunk_id → passage_id
        """
        all_chunks: list[ChunkRecord] = []
        chunk_to_passage: dict[str, str] = {}

        for pid, text in passages:
            for chunk in self.chunk_passage(pid, text):
                all_chunks.append(chunk)
                chunk_to_passage[chunk.chunk_id] = chunk.passage_id

        return all_chunks, chunk_to_passage
