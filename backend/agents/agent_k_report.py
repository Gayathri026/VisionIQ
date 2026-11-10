import os
import time
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from backend.config import settings

class ReportGeneratorAgent:
    """
    Agent K: Engineering PDF Report Generation Module
    Uses ReportLab to produce a formal, verifiable engineering inspection report
    containing annotated visual evidence, finding tables, routing traces,
    RAG citations, root cause hypotheses, and reproducibility metadata.
    """
    def __init__(self):
        self.agent_name = "report"
        self.version = "VisionIQ-PDF-Engine-v1.0"

    def generate_pdf(
        self,
        session_id: str,
        query: str,
        input_path: str,
        annotated_path: str,
        findings: List[Dict[str, Any]],
        traces: List[Dict[str, Any]],
        rag_references: List[Dict[str, Any]],
        rca_result: Dict[str, Any] = None,
        recommendation_result: Dict[str, Any] = None,
        evidence_support_score: float = 1.0,
        total_latency_ms: float = 0.0
    ) -> str:
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)
        pdf_filename = f"VisionIQ_Inspection_Report_{session_id[:8]}.pdf"
        output_pdf_path = os.path.join(settings.REPORTS_DIR, pdf_filename)

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Technical Palette
        PRIMARY = colors.HexColor("#1A202C")     # Graphite Dark
        ACCENT = colors.HexColor("#2B6CB0")      # Technical Steel Blue
        BORDER_GRAY = colors.HexColor("#E2E8F0") # Clean Light Gray
        BG_LIGHT = colors.HexColor("#F7FAFC")    # Warm White
        TEXT_MUTED = colors.HexColor("#4A5568")  # Muted Gray
        WARNING_COLOR = colors.HexColor("#DD6B20")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=PRIMARY
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=TEXT_MUTED
        )
        h1_style = ParagraphStyle(
            "SectionH1",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceAfter=6,
            spaceBefore=10
        )
        body_style = ParagraphStyle(
            "BodyTechnical",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=PRIMARY
        )
        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=TEXT_MUTED,
            alignment=1
        )

        elements = []

        # Header Title
        elements.append(Paragraph("VisionIQ — Visual Intelligence Inspection Report", title_style))
        elements.append(Paragraph(
            "Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration & Cost-Aware Routing<br/>"
            "Academic Evaluation & Industrial Provenance Record | Dr. Mahalingam College of Engineering and Technology",
            subtitle_style
        ))
        elements.append(Spacer(1, 10))

        # Metadata Box
        meta_data = [
            [
                Paragraph(f"<b>Report ID:</b> {session_id}", body_style),
                Paragraph(f"<b>Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S UTC')}", body_style)
            ],
            [
                Paragraph(f"<b>Target Query:</b> \"{query}\"", body_style),
                Paragraph(f"<b>Evidence Support Score:</b> {evidence_support_score:.2f} / 1.00", body_style)
            ],
            [
                Paragraph(f"<b>Total Latency:</b> {total_latency_ms:.1f} ms", body_style),
                Paragraph(f"<b>Software Version:</b> VisionIQ v1.0.0 (LangGraph + PyTorch)", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # Section 1: Visual Evidence
        elements.append(Paragraph("1. Visual Evidence Verification", h1_style))
        img_row = []
        if input_path and os.path.exists(input_path):
            try:
                img_row.append([RLImage(input_path, width=3.4*inch, height=2.4*inch)])
            except Exception:
                img_row.append([Paragraph("Original image format incompatible with PDF embed", caption_style)])
        else:
            img_row.append([Paragraph("Original Image Not Available", caption_style)])

        if annotated_path and os.path.exists(annotated_path):
            try:
                img_row.append([RLImage(annotated_path, width=3.4*inch, height=2.4*inch)])
            except Exception:
                img_row.append([Paragraph("Annotated image format incompatible with PDF embed", caption_style)])
        else:
            img_row.append([Paragraph("Annotated View Not Available", caption_style)])

        if len(img_row) == 2:
            img_table = Table([[img_row[0][0], img_row[1][0]]], colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ]))
            elements.append(img_table)
            elements.append(Table([[
                Paragraph("Figure 1(a): Source Visual Input", caption_style),
                Paragraph("Figure 1(b): VisionIQ Annotated Evidence Overlays", caption_style)
            ]], colWidths=[270, 270]))
        elements.append(Spacer(1, 10))

        # Section 2: Findings Table
        elements.append(Paragraph("2. Detected Entities & Specialized Findings", h1_style))
        finding_rows = [[
            Paragraph("<b>UID</b>", body_style),
            Paragraph("<b>Agent</b>", body_style),
            Paragraph("<b>Category / Classification</b>", body_style),
            Paragraph("<b>Confidence / Anomaly</b>", body_style),
            Paragraph("<b>Observed Evidence Description</b>", body_style),
            Paragraph("<b>Human Review</b>", body_style)
        ]]

        for f in findings[:8]:  # Display top 8 findings cleanly
            conf_str = f"{f['detection_confidence']*100:.1f}%" if f.get("detection_confidence") else (
                f"Anomaly {f['anomaly_score']:.2f}" if f.get("anomaly_score") else "N/A"
            )
            review_str = "REQUIRED" if f.get("human_review_required") else "Verified"
            finding_rows.append([
                Paragraph(f.get("finding_uid", "F-00"), body_style),
                Paragraph(f.get("agent_source", "").upper(), body_style),
                Paragraph(f.get("label", "Unknown"), body_style),
                Paragraph(conf_str, body_style),
                Paragraph(f.get("observed_evidence", "")[:120] + "...", body_style),
                Paragraph(review_str, body_style)
            ])

        if len(finding_rows) == 1:
            finding_rows.append([Paragraph("No discrete object/defect findings recorded", body_style), "", "", "", "", ""])

        finding_table = Table(finding_rows, colWidths=[55, 55, 110, 75, 185, 60])
        finding_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(finding_table)
        elements.append(Spacer(1, 12))

        # Section 3: Cost-Aware Dynamic Routing Trace Table
        elements.append(Paragraph("3. Cost-Aware Agent Routing & Execution Trace (Equation 1)", h1_style))
        trace_rows = [[
            Paragraph("<b>Agent</b>", body_style),
            Paragraph("<b>Relevance R</b>", body_style),
            Paragraph("<b>Capability E</b>", body_style),
            Paragraph("<b>Cost C</b>", body_style),
            Paragraph("<b>Latency L</b>", body_style),
            Paragraph("<b>Score S</b>", body_style),
            Paragraph("<b>Decision</b>", body_style),
            Paragraph("<b>Runtime</b>", body_style)
        ]]

        for t in traces:
            t_score = t.get("final_routing_score", 0.0)
            trace_rows.append([
                Paragraph(t.get("agent_name", ""), body_style),
                Paragraph(f"{t.get('relevance_score', 0.0):.2f}", body_style),
                Paragraph(f"{t.get('evidence_capability', 0.0):.2f}", body_style),
                Paragraph(f"{t.get('normalized_cost', 0.0):.2f}", body_style),
                Paragraph(f"{t.get('expected_latency', 0.0):.2f}", body_style),
                Paragraph(f"<b>{t_score:.3f}</b>", body_style),
                Paragraph(t.get("decision", ""), body_style),
                Paragraph(f"{t.get('actual_latency_ms', 0.0):.1f} ms", body_style)
            ])

        trace_table = Table(trace_rows, colWidths=[100, 60, 60, 55, 55, 60, 90, 60])
        trace_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 1, BORDER_GRAY),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(trace_table)
        elements.append(Spacer(1, 12))

        # Section 4: RAG Grounding & Engineering Standards Citations
        if rag_references:
            elements.append(Paragraph("4. RAG Knowledge Grounding & Citations", h1_style))
            for idx, r in enumerate(rag_references):
                p_text = (
                    f"<b>[{idx+1}] {r['document_title']}</b> ({r['section_reference']}) "
                    f"— <i>Relevance: {r['similarity_score']*100:.1f}%</i><br/>"
                    f"\"{r['excerpt']}\"<br/>"
                    f"<font color='#2B6CB0'>Source: {r.get('source_uri', 'Internal Technical Archive')}</font>"
                )
                elements.append(Paragraph(p_text, body_style))
                elements.append(Spacer(1, 4))
            elements.append(Spacer(1, 8))

        # Section 5: RCA & Recommendations
        if rca_result and rca_result.get("ranked_hypotheses"):
            elements.append(Paragraph("5. Root Cause Hypotheses & Diagnostic Verification", h1_style))
            for h in rca_result["ranked_hypotheses"]:
                elements.append(Paragraph(
                    f"<b>Rank {h['rank']} Hypothesis ({h['estimated_cause_probability']*100:.0f}% likelihood):</b> {h['hypothesis']}<br/>"
                    f"• <i>Observed Evidence:</i> {h['supporting_visual_evidence']}<br/>"
                    f"• <i>Caveat:</i> {h['caveats']}",
                    body_style
                ))
                elements.append(Spacer(1, 4))

        if recommendation_result and recommendation_result.get("immediate_corrective_actions"):
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("6. Corrective & Preventive Action (CAPA) Plan", h1_style))
            for act in recommendation_result["immediate_corrective_actions"]:
                elements.append(Paragraph(
                    f"• <b>[{act['priority']}] {act['title']}</b>: {act['description']} "
                    f"(<i>Standard: {act['linked_standard']} | Certified Sign-off: {'YES' if act['requires_human_signoff'] else 'NO'}</i>)",
                    body_style
                ))
                elements.append(Spacer(1, 4))

        # Section 7: Reproducibility & Provenance
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("7. Academic Provenance & Reproducibility Parameters", h1_style))
        repro_text = (
            f"<b>Framework:</b> VisionIQ v1.0.0 | <b>Graph Engine:</b> LangGraph 0.2.20 | "
            f"<b>Detector:</b> YOLOv11n (Ultralytics 8.3.0) | <b>OCR:</b> EasyOCR 1.7.2 | "
            f"<b>Routing Weights:</b> alpha={settings.ROUTING_ALPHA}, beta={settings.ROUTING_BETA}, "
            f"gamma={settings.ROUTING_GAMMA}, delta={settings.ROUTING_DELTA}, tau={settings.ROUTING_THRESHOLD} | "
            f"<b>Target Journal:</b> IEEE Access Submission Draft | <b>Institution:</b> Dr. Mahalingam College of Engineering and Technology."
        )
        elements.append(Paragraph(repro_text, caption_style))

        # Build document
        doc.build(elements)
        return output_pdf_path

def run_report_agent(state: dict) -> dict:
    session_id = state.get("session_id", "sess-default")
    query = state.get("user_query", "")
    input_path = state.get("image_path", "")
    annotated_path = state.get("annotated_path", "")
    findings = state.get("findings", [])
    traces = state.get("execution_log", [])
    rag_refs = state.get("rag_references", [])
    rca = state.get("rca_result")
    recom = state.get("recommendation_result")
    evidence_score = state.get("evidence_support_score", 1.0)
    total_latency = state.get("total_latency_ms", 0.0)

    agent = ReportGeneratorAgent()
    pdf_path = agent.generate_pdf(
        session_id=session_id,
        query=query,
        input_path=input_path,
        annotated_path=annotated_path,
        findings=findings,
        traces=traces,
        rag_references=rag_refs,
        rca_result=rca,
        recommendation_result=recom,
        evidence_support_score=evidence_score,
        total_latency_ms=total_latency
    )
    return {"pdf_report_path": pdf_path}
