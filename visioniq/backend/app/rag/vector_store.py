import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import settings
from backend.app.core.logging import logger

class HybridVectorStore:
    """
    Persistent Hybrid Vector Store combining TF-IDF n-gram subword embeddings
    with BM25 lexical matching for precise industrial standards retrieval.
    """
    def __init__(self, index_path: Optional[str] = None):
        self.index_dir = os.path.join(settings.BASE_DIR, "storage", "rag")
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = index_path or os.path.join(self.index_dir, "rag_index.json")

        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        # BM25 statistics
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.k1: float = 1.5
        self.b: float = 0.75

        self.load_index()

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> int:
        """Adds new chunks to the store and updates the vector representations."""
        if not new_chunks:
            return 0

        existing_ids = {c["chunk_id"] for c in self.chunks}
        filtered_new = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
        if not filtered_new:
            return 0

        self.chunks.extend(filtered_new)
        self._rebuild_index()
        self.save_index()
        return len(filtered_new)

    def remove_document(self, doc_id: str) -> int:
        """Removes all chunks associated with a specific document ID."""
        initial_len = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("doc_id") != doc_id]
        removed = initial_len - len(self.chunks)
        if removed > 0:
            self._rebuild_index()
            self.save_index()
        return removed

    def _rebuild_index(self):
        """Rebuilds the vectorizer and BM25 term frequency structures."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_lens = []
            self.avg_doc_len = 0.0
            return

        # Index title + section + content together for comprehensive coverage
        corpus = [f"{c['document_title']} {c['section_reference']} {c['content']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            token_pattern=r"(?u)\b[\w\.\-]+\b"  # Retain standard numbers like 1926.100, Z89.1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Compute BM25 doc lengths
        self.doc_lens = [len(c["content"].lower().split()) for c in self.chunks]
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0

    def _bm25_score(self, query_tokens: List[str], chunk_idx: int) -> float:
        """Computes BM25 lexical matching score between query tokens and a chunk."""
        if not self.chunks or chunk_idx >= len(self.chunks):
            return 0.0

        content_tokens = self.chunks[chunk_idx]["content"].lower().split()
        doc_len = self.doc_lens[chunk_idx] if chunk_idx < len(self.doc_lens) else len(content_tokens)
        N = len(self.chunks)

        stopwords = {"what", "is", "the", "are", "under", "for", "in", "to", "and", "of", "a", "an", "on", "with"}
        informative_tokens = [t for t in query_tokens if t not in stopwords]
        if not informative_tokens:
            informative_tokens = query_tokens

        score = 0.0
        for token in informative_tokens:
            tf = content_tokens.count(token)
            if tf == 0:
                continue

            # Document frequency
            df = sum(1 for c in self.chunks if token in c["content"].lower())
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
            score += idf * (numerator / max(0.001, denominator))

        return min(1.0, score / max(1.0, 5.0 * len(informative_tokens)))

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        relevance_threshold: float = 0.30,
        dense_weight: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid semantic retrieval and returns ranked chunks above relevance threshold.
        """
        if not self.chunks or not self.vectorizer or not query_text.strip():
            return []

        # 1. Dense TF-IDF Cosine Similarity
        q_vec = self.vectorizer.transform([query_text])
        cos_sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # 2. BM25 Lexical Score
        q_tokens = [t.strip(",.?!:;\"'") for t in query_text.lower().split() if t.strip(",.?!:;\"'")]
        stopwords = {"what", "is", "the", "are", "under", "for", "in", "to", "and", "of", "a", "an", "on", "with", "standard", "standards", "required", "workers", "inspection"}
        salient_tokens = [t for t in q_tokens if t not in stopwords]

        results = []
        for idx, chunk in enumerate(self.chunks):
            dense_score = float(cos_sims[idx])
            bm25_score = self._bm25_score(q_tokens, idx)

            # Boost matches of salient keywords in title or section reference
            title_text = f"{chunk['document_title']} {chunk['section_reference']}".lower()
            salient_matches = sum(1 for t in salient_tokens if t in title_text)
            title_boost = 0.15 * min(2, salient_matches)

            hybrid_score = round(
                min(0.99, dense_weight * dense_score + (1.0 - dense_weight) * bm25_score + title_boost),
                4
            )

            if hybrid_score >= relevance_threshold:
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "document_title": chunk["document_title"],
                    "section_reference": chunk["section_reference"],
                    "source_uri": chunk.get("source_uri"),
                    "page_number": chunk.get("page_number"),
                    "excerpt": chunk["content"],
                    "similarity_score": hybrid_score,
                    "dense_score": round(dense_score, 4),
                    "lexical_score": round(bm25_score, 4)
                })

        # Sort descending by hybrid similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def save_index(self):
        """Serializes indexed chunks to disk."""
        data = {
            "chunks": self.chunks,
            "total_chunks": len(self.chunks)
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_index(self):
        """Loads serialized chunks from disk and initializes vectorizer."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self._rebuild_index()
            except Exception as e:
                logger.warning(f"Unable to load RAG index from {self.index_path}: {e}")

hybrid_vector_store = HybridVectorStore()
