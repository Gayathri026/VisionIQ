import os
import shutil
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.rag.rag_service import rag_service
from backend.app.rag.synthesizer import evidence_synthesizer

router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the mandatory hard hat standard under OSHA?"})
    top_k: int = Field(3, ge=1, le=10)
    relevance_threshold: float = Field(0.30, ge=0.0, le=1.0)
    dense_weight: float = Field(0.60, ge=0.0, le=1.0)

class CitationValidationRequest(BaseModel):
    citations: List[Dict[str, Any]]

@router.get("/rag/documents", summary="List All Indexed Knowledge Base Documents")
async def list_rag_documents():
    """Returns metadata for all documents currently indexed in the persistent hybrid vector store."""
    docs = rag_service.list_documents()
    total_chunks = len(rag_service.store.chunks)
    return {
        "status": "success",
        "total_documents": len(docs),
        "total_indexed_chunks": total_chunks,
        "documents": docs
    }

@router.post("/rag/ingest", summary="Ingest and Index New Document (PDF, MD, TXT, HTML)")
async def ingest_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source_uri: Optional[str] = Form(None)
):
    """
    Safely ingests, parses, chunks, and indexes an engineering document or regulatory standard.
    Enforces prompt injection neutralization and extracts section/page metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload payload.")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {".pdf", ".md", ".txt", ".html", ".htm", ".markdown"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(allowed_exts))}"
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded document is empty (0 bytes).")

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Document exceeds 20MB limit.")

    # Save to knowledge base storage
    kb_dir = os.path.join(settings.BASE_DIR, "visioniq", "knowledge_base")
    os.makedirs(kb_dir, exist_ok=True)
    safe_name = f"doc_{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(kb_dir, safe_name)

    with open(save_path, "wb") as f_out:
        f_out.write(content)

    try:
        res = rag_service.ingest_file(
            file_path=save_path,
            source_uri=source_uri or f"file:///{save_path.replace(os.sep, '/')}",
            title=title or file.filename
        )
    except Exception as e:
        logger.error(f"Failed to ingest document {file.filename}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Document parsing/indexing failure: {str(e)}")

    return {
        "status": "success",
        "message": f"Successfully indexed '{file.filename}'",
        "chunks_indexed": res.get("chunks_added", 0),
        "document_id": res.get("document_id"),
        "result": res
    }

@router.post("/rag/query", summary="Query RAG Knowledge Store")
async def query_rag(req: RAGQueryRequest):
    """
    Executes hybrid semantic search over indexed engineering standards and returns
    ranked passages with scores, citations, and epistemological layer separation.
    """
    passages = rag_service.query(
        query_text=req.query,
        top_k=req.top_k,
        relevance_threshold=req.relevance_threshold
    )

    # Generate synthesized evidence report
    synth = evidence_synthesizer.synthesize(
        query=req.query,
        findings=[],
        rag_references=passages,
        detected_intent="Technical Grounding Search"
    )

    return {
        "status": "success",
        "query": req.query,
        "total_retrieved": len(passages),
        "total_found": len(passages),
        "passages": passages,
        "results": passages,
        "synthesis": synth
    }

@router.post("/rag/validate-citations", summary="Validate Citation Provenance")
async def validate_citations(req: CitationValidationRequest):
    """
    Verifies that cited standard numbers and document titles correspond to
    actual excerpts in the persistent knowledge store.
    """
    res = rag_service.validate_citations(req.citations)
    return {
        "status": "success",
        "validation": res
    }

@router.delete("/rag/documents/{doc_id}", summary="Remove Document from Knowledge Store")
async def delete_document(doc_id: str):
    """Removes a document and all its chunks from the persistent vector index."""
    removed = rag_service.store.remove_document(doc_id)
    if doc_id in rag_service.indexed_documents:
        del rag_service.indexed_documents[doc_id]

    if removed == 0:
        raise HTTPException(status_code=404, detail="Document ID not found in index.")

    return {
        "status": "success",
        "message": f"Removed document '{doc_id}' ({removed} chunks deleted)."
    }
