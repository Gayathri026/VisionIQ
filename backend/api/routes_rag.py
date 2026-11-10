from fastapi import APIRouter
from backend.agents.agent_h_rag import OFFICIAL_DOCUMENTS_CORPUS

router = APIRouter()

@router.get("/rag/documents")
async def list_rag_documents():
    return {
        "status": "success",
        "total": len(OFFICIAL_DOCUMENTS_CORPUS),
        "documents": [
            {
                "title": doc["document_title"],
                "section": doc["section_reference"],
                "source_uri": doc["source_uri"],
                "keywords": doc["keywords"],
                "excerpt_preview": doc["content"][:160] + "..."
            }
            for doc in OFFICIAL_DOCUMENTS_CORPUS
        ]
    }
