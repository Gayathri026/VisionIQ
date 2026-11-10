import os
import re
import uuid
from typing import List, Dict, Any, Optional
from html.parser import HTMLParser

class SimpleHTMLTextExtractor(HTMLParser):
    """Safely extracts text and section headers from HTML without external dependencies."""
    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.sections: List[Dict[str, Any]] = []
        self.current_heading = "General"
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._in_script = True
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.text_parts.append("\n\n### ")

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self._in_script = False
        elif tag in ('p', 'div', 'br', 'li'):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self._in_script and not self._in_style:
            clean = data.strip()
            if clean:
                self.text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self.text_parts).strip()

def sanitize_text_for_prompt_injection(text: str) -> str:
    """
    Sanitizes raw document text to neutralize prompt injection attacks.
    Prevents untrusted documents from hijacking the LLM or agent policy.
    """
    # Replace dangerous directive patterns with inert markers
    injection_patterns = [
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)disregard\s+(all\s+)?prior\s+guidelines",
        r"(?i)system\s+prompt\s*:",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)new\s+operating\s+rule\s*:",
        r"(?i)override\s+system\s+policy",
        r"(?i)execute\s+the\s+following\s+code",
        r"(?i)<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>"
    ]
    sanitized = text
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[FILTERED_INSTRUCTION]", sanitized)
    return sanitized

class DocumentLoader:
    """
    Ingests and extracts structured text and metadata from PDF, Markdown, Plain Text, and HTML files.
    Extracts page or section references and enforces prompt injection safeguards.
    """
    def load_file(self, file_path: str, source_uri: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found at {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        doc_id = str(uuid.uuid4())
        doc_title = title or os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()

        if ext == ".pdf":
            return self._load_pdf(file_path, doc_id, doc_title, source_uri)
        elif ext in (".md", ".markdown"):
            return self._load_markdown(file_path, doc_id, doc_title, source_uri, has_custom_title=bool(title))
        elif ext in (".txt", ".text"):
            return self._load_txt(file_path, doc_id, doc_title, source_uri)
        elif ext in (".html", ".htm"):
            return self._load_html(file_path, doc_id, doc_title, source_uri)
        else:
            raise ValueError(f"Unsupported document format '{ext}'. Supported: .pdf, .md, .txt, .html")

    def _load_pdf(self, file_path: str, doc_id: str, doc_title: str, source_uri: Optional[str]) -> Dict[str, Any]:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages_data = []

        full_text_list = []
        for p_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            sanitized_text = sanitize_text_for_prompt_injection(page_text)
            if sanitized_text.strip():
                pages_data.append({
                    "page_number": p_idx + 1,
                    "section_reference": f"Page {p_idx + 1}",
                    "text": sanitized_text
                })
                full_text_list.append(sanitized_text)

        return {
            "doc_id": doc_id,
            "title": doc_title,
            "format": "pdf",
            "file_path": file_path,
            "source_uri": source_uri or f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}",
            "total_pages": len(reader.pages),
            "sections": pages_data,
            "full_text": "\n\n".join(full_text_list)
        }

    def _load_markdown(self, file_path: str, doc_id: str, doc_title: str, source_uri: Optional[str], has_custom_title: bool = False) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        sanitized_text = sanitize_text_for_prompt_injection(raw_text)
        sections = []
        lines = sanitized_text.splitlines()

        # Extract primary H1 as document title if caller did not supply custom title
        if not has_custom_title:
            first_h1 = None
            for line in lines:
                if line.startswith("# ") and not line.startswith("##"):
                    first_h1 = line.lstrip("#").strip()
                    break
            if first_h1:
                doc_title = first_h1

        current_section = "Overview"
        current_buffer = []

        for line in lines:
            if line.startswith("#"):
                if current_buffer:
                    sec_text = "\n".join(current_buffer).strip()
                    if sec_text:
                        sections.append({
                            "page_number": None,
                            "section_reference": current_section,
                            "text": sec_text
                        })
                    current_buffer = []
                current_section = line.lstrip("#").strip()
            else:
                current_buffer.append(line)

        if current_buffer:
            sec_text = "\n".join(current_buffer).strip()
            if sec_text:
                sections.append({
                    "page_number": None,
                    "section_reference": current_section,
                    "text": sec_text
                })

        return {
            "doc_id": doc_id,
            "title": doc_title,
            "format": "markdown",
            "file_path": file_path,
            "source_uri": source_uri or f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}",
            "total_pages": 1,
            "sections": sections,
            "full_text": sanitized_text
        }

    def _load_txt(self, file_path: str, doc_id: str, doc_title: str, source_uri: Optional[str]) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        sanitized_text = sanitize_text_for_prompt_injection(raw_text)
        paragraphs = [p.strip() for p in sanitized_text.split("\n\n") if p.strip()]
        sections = [
            {
                "page_number": None,
                "section_reference": f"Section {i + 1}",
                "text": p
            }
            for i, p in enumerate(paragraphs)
        ]

        return {
            "doc_id": doc_id,
            "title": doc_title,
            "format": "text",
            "file_path": file_path,
            "source_uri": source_uri or f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}",
            "total_pages": 1,
            "sections": sections if sections else [{"page_number": None, "section_reference": "General", "text": sanitized_text}],
            "full_text": sanitized_text
        }

    def _load_html(self, file_path: str, doc_id: str, doc_title: str, source_uri: Optional[str]) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        parser = SimpleHTMLTextExtractor()
        parser.feed(html_content)
        extracted = parser.get_text()
        sanitized_text = sanitize_text_for_prompt_injection(extracted)

        return {
            "doc_id": doc_id,
            "title": doc_title,
            "format": "html",
            "file_path": file_path,
            "source_uri": source_uri or f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}",
            "total_pages": 1,
            "sections": [{"page_number": None, "section_reference": "Document Body", "text": sanitized_text}],
            "full_text": sanitized_text
        }

document_loader = DocumentLoader()
