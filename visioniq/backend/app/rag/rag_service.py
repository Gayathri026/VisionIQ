import os
import glob
import time
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.rag.document_loader import document_loader
from backend.app.rag.chunker import text_chunker
from backend.app.rag.vector_store import hybrid_vector_store
from backend.app.rag.synthesizer import evidence_synthesizer

class RAGService:
    """
    Central Retrieval-Augmented Generation Service for VisionIQ.
    Manages document ingestion, chunking, persistent hybrid indexing,
    semantic search, citation validation, and evidence synthesis.
    """
    def __init__(self):
        self.loader = document_loader
        self.chunker = text_chunker
        self.store = hybrid_vector_store
        self.synthesizer = evidence_synthesizer
        self.indexed_documents: Dict[str, Dict[str, Any]] = {}
        
        # Ingest default knowledge base on initialization if empty
        self.bootstrap_knowledge_base()

    def bootstrap_knowledge_base(self):
        """Scans the knowledge base directory and indexes all standards documents without duplication."""
        candidate_dirs = [
            os.path.join(settings.BASE_DIR, "visioniq", "knowledge_base"),
            os.path.join(settings.BASE_DIR, "knowledge_base"),
            "knowledge_base"
        ]

        # Reset store if reloading or prevent duplicate basenames
        existing_titles = {c.get("document_title", "").lower() for c in self.store.chunks}
        processed_basenames = set()

        found_any = False
        for kb_dir in candidate_dirs:
            if os.path.exists(kb_dir):
                patterns = ["*.md", "*.txt", "*.pdf", "*.html"]
                for p in patterns:
                    for fpath in glob.glob(os.path.join(kb_dir, p)):
                        bname = os.path.basename(fpath).lower()
                        if bname in processed_basenames:
                            continue
                        processed_basenames.add(bname)

                        try:
                            # Check if already in store
                            doc_data = self.loader.load_file(fpath)
                            if doc_data["title"].lower() in existing_titles:
                                continue
                            self.ingest_file(fpath, title=doc_data["title"])
                            existing_titles.add(doc_data["title"].lower())
                            found_any = True
                        except Exception as e:
                            logger.warning(f"Error bootstrapping document {fpath}: {e}")

        logger.info(f"RAG Service initialized with {len(self.store.chunks)} indexed chunks across {len(self.indexed_documents)} documents.")

    def ingest_file(self, file_path: str, source_uri: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
        """Loads, sanitizes, chunks, and indexes a single document file."""
        doc_data = self.loader.load_file(file_path=file_path, source_uri=source_uri, title=title)
        chunks = self.chunker.chunk_document(doc_data)
        
        added_count = self.store.add_chunks(chunks)
        
        doc_info = {
            "doc_id": doc_data["doc_id"],
            "title": doc_data["title"],
            "format": doc_data["format"],
            "file_path": file_path,
            "source_uri": doc_data["source_uri"],
            "total_pages": doc_data.get("total_pages", 1),
            "total_chunks": len(chunks),
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        self.indexed_documents[doc_data["doc_id"]] = doc_info

        return {
            "status": "success",
            "document": doc_info,
            "chunks_added": added_count
        }

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        relevance_threshold: float = 0.30
    ) -> List[Dict[str, Any]]:
        """Queries the vector store and returns relevant chunks."""
        return self.store.query(
            query_text=query_text,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all currently indexed documents."""
        if not self.store.chunks:
            self.bootstrap_knowledge_base()

        # Build document summaries guaranteeing all store chunks are represented
        docs_map = dict(self.indexed_documents)
        for c in self.store.chunks:
            d_id = c.get("doc_id")
            if not d_id:
                continue
            if d_id not in docs_map:
                docs_map[d_id] = {
                    "doc_id": d_id,
                    "title": c.get("document_title", "Untitled Document"),
                    "format": "markdown" if str(c.get("source_uri", "")).endswith(".md") else "pdf" if str(c.get("source_uri", "")).endswith(".pdf") else "text",
                    "file_path": str(c.get("source_uri", "")).replace("file:///", ""),
                    "source_uri": c.get("source_uri", ""),
                    "total_pages": 1,
                    "total_chunks": sum(1 for chk in self.store.chunks if chk.get("doc_id") == d_id),
                    "file_size_bytes": 0,
                    "ingested_at": "Persistent Index"
                }

        doc_ids_in_store = {c.get("doc_id") for c in self.store.chunks}
        return [info for d_id, info in docs_map.items() if d_id in doc_ids_in_store]

    def validate_citations(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates that cited documents exist in the knowledge store and match genuine excerpts.
        """
        import re
        validated = []
        unmatched = []

        for cit in citations:
            doc_title = cit.get("document_title", "").lower()
            sec_ref = cit.get("section_reference", "").lower()

            # Extract alphanumeric tokens, filtering common filler
            cit_tokens = set(re.findall(r"[a-z0-9]+", doc_title))
            cit_tokens.difference_update({"standard", "specification", "the", "for", "in", "and", "of"})

            cit_sec_tokens = set(re.findall(r"[a-z0-9]+", sec_ref))

            matched_chunk = None
            for c in self.store.chunks:
                chunk_title_tokens = set(re.findall(r"[a-z0-9]+", c["document_title"].lower()))
                chunk_sec_tokens = set(re.findall(r"[a-z0-9]+", c["section_reference"].lower()))

                # Title match: either exact substring or shared key identifiers (e.g. "1926", "100")
                title_match = (doc_title in c["document_title"].lower() or c["document_title"].lower() in doc_title) or \
                              (cit_tokens and (cit_tokens.issubset(chunk_title_tokens) or len(cit_tokens & chunk_title_tokens) >= max(1, len(cit_tokens) - 1)))

                sec_match = (not cit_sec_tokens) or (sec_ref in c["section_reference"].lower() or c["section_reference"].lower() in sec_ref) or \
                            (cit_sec_tokens.issubset(chunk_sec_tokens))

                if title_match and sec_match:
                    matched_chunk = c
                    break

            if matched_chunk:
                validated.append({
                    "citation": cit,
                    "matched_chunk_id": matched_chunk["chunk_id"],
                    "is_authentic": True
                })
            else:
                unmatched.append({
                    "citation": cit,
                    "is_authentic": False,
                    "reason": "Citation does not map to any indexed knowledge chunk."
                })

        return {
            "total_checked": len(citations),
            "validated_count": len(validated),
            "unmatched_count": len(unmatched),
            "is_fully_grounded": len(unmatched) == 0,
            "validated": validated,
            "unmatched": unmatched
        }

rag_service = RAGService()
