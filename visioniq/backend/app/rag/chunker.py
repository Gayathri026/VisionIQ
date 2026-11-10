import re
from typing import List, Dict, Any

class TextChunker:
    """
    Splits structured documents into indexed chunks with source provenance,
    preserving section titles, page references, and context overlap.
    """
    def __init__(self, target_chunk_size: int = 500, chunk_overlap: int = 100):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        doc_id = doc_data["doc_id"]
        doc_title = doc_data["title"]
        source_uri = doc_data.get("source_uri", "")

        chunk_counter = 1

        for sec in doc_data.get("sections", []):
            sec_ref = sec.get("section_reference", "General")
            page_num = sec.get("page_number")
            sec_text = sec.get("text", "").strip()

            if not sec_text:
                continue

            # If section text fits in one chunk, keep it intact
            if len(sec_text) <= self.target_chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_id[:8]}_c{chunk_counter:03d}",
                    "doc_id": doc_id,
                    "document_title": doc_title,
                    "section_reference": sec_ref,
                    "page_number": page_num,
                    "source_uri": source_uri,
                    "content": sec_text,
                    "char_count": len(sec_text),
                    "token_count": len(sec_text.split())
                })
                chunk_counter += 1
                continue

            # Otherwise, split across sentences or paragraphs with overlap
            sentences = re.split(r'(?<=[.!?])\s+', sec_text)
            curr_chunk = []
            curr_len = 0

            for s in sentences:
                s_len = len(s)
                if curr_len + s_len > self.target_chunk_size and curr_chunk:
                    chunk_str = " ".join(curr_chunk).strip()
                    chunks.append({
                        "chunk_id": f"{doc_id[:8]}_c{chunk_counter:03d}",
                        "doc_id": doc_id,
                        "document_title": doc_title,
                        "section_reference": sec_ref,
                        "page_number": page_num,
                        "source_uri": source_uri,
                        "content": chunk_str,
                        "char_count": len(chunk_str),
                        "token_count": len(chunk_str.split())
                    })
                    chunk_counter += 1

                    # Retain last few sentences for overlap
                    overlap_sentences = []
                    overlap_len = 0
                    for prev_s in reversed(curr_chunk):
                        if overlap_len + len(prev_s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, prev_s)
                            overlap_len += len(prev_s)
                        else:
                            break
                    curr_chunk = overlap_sentences
                    curr_len = overlap_len

                curr_chunk.append(s)
                curr_len += s_len

            if curr_chunk:
                chunk_str = " ".join(curr_chunk).strip()
                if chunk_str:
                    chunks.append({
                        "chunk_id": f"{doc_id[:8]}_c{chunk_counter:03d}",
                        "doc_id": doc_id,
                        "document_title": doc_title,
                        "section_reference": sec_ref,
                        "page_number": page_num,
                        "source_uri": source_uri,
                        "content": chunk_str,
                        "char_count": len(chunk_str),
                        "token_count": len(chunk_str.split())
                    })
                    chunk_counter += 1

        return chunks

text_chunker = TextChunker()
