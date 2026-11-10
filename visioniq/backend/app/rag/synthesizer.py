from typing import List, Dict, Any, Optional

class EvidenceSynthesizer:
    """
    Rigorously synthesizes multi-agent visual findings with retrieved engineering standards,
    strictly separating epistemological layers according to IEEE Access research standards:
    1. Direct Visual Observations
    2. Model Predictions & Confidence Interpretations
    3. Retrieved Technical Standards & Regulations
    4. Inferred Causal Hypotheses (explicitly qualified as hypotheses)
    5. Actionable Recommendations & NDT Verification Steps
    """
    def synthesize(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        rag_references: List[Dict[str, Any]],
        detected_intent: Optional[str] = None
    ) -> Dict[str, Any]:
        observations = []
        model_predictions = []
        causal_hypotheses = []
        recommendations = []

        for f in findings:
            lbl = f.get("label", "Detection")
            obs = f.get("observed_evidence", "")
            conf = f.get("detection_confidence")
            score_desc = f.get("score_interpretation")
            model_info = f"{f.get('model_name', 'Model')} ({f.get('model_version', 'v1.0')})"
            
            # Layer 1: Visual observations
            observations.append(obs if obs else f"Visual observation of {lbl} at bbox {f.get('bbox')}.")

            # Layer 2: Model predictions
            conf_str = f"{conf*100:.1f}%" if conf is not None else "N/A"
            model_predictions.append(f"{model_info} detected {lbl} with confidence {conf_str}. {score_desc or ''}")

            # Layer 4: Inferred causal hypotheses
            hyp = f.get("root_cause_hypothesis")
            alts = f.get("alternative_explanations")
            if hyp:
                causal_hypotheses.append({
                    "finding_uid": f.get("finding_uid", "F"),
                    "hypothesis": hyp,
                    "alternative_explanations": alts,
                    "epistemic_status": "Hypothesis pending physical verification"
                })

            # Layer 5: Recommendations
            rec = f.get("recommendation")
            proc = f.get("verification_procedure")
            if rec or proc:
                recommendations.append({
                    "finding_uid": f.get("finding_uid", "F"),
                    "corrective_action": rec or "Conduct engineering review.",
                    "verification_procedure": proc or "Visual inspection."
                })

        # Layer 3: Retrieved Technical Standards
        retrieved_standards = [
            {
                "document_title": r.get("document_title"),
                "section_reference": r.get("section_reference"),
                "similarity_score": r.get("similarity_score"),
                "excerpt": r.get("excerpt"),
                "source_uri": r.get("source_uri")
            }
            for r in rag_references
        ]

        # Overall synthesis text
        synthesis_lines = [
            f"VisionIQ Multi-Agent Analysis Report for query: '{query}'",
            f"Detected Intent: {detected_intent or 'Multi-Agent Inspection'}\n",
            "1. DIRECT VISUAL OBSERVATIONS:"
        ]
        if observations:
            synthesis_lines.extend(f"  - {o}" for o in observations[:4])
        else:
            synthesis_lines.append("  - No visual anomalies detected.")

        synthesis_lines.append("\n2. MODEL PREDICTIONS & PROVENANCE:")
        if model_predictions:
            synthesis_lines.extend(f"  - {mp}" for mp in model_predictions[:4])
        else:
            synthesis_lines.append("  - Baseline vision model executed.")

        synthesis_lines.append("\n3. RETRIEVED REGULATORY STANDARDS & TECHNICAL GROUNDING:")
        if retrieved_standards:
            synthesis_lines.extend(
                f"  - [{r['document_title']} | {r['section_reference']}] (Match: {r['similarity_score']*100:.1f}%): \"{r['excerpt'][:140]}...\""
                for r in retrieved_standards
            )
        else:
            synthesis_lines.append("  - No external standards cited.")

        synthesis_lines.append("\n4. INFERRED CAUSAL HYPOTHESES (NOT GROUND TRUTH):")
        if causal_hypotheses:
            synthesis_lines.extend(
                f"  - [{h['finding_uid']}] {h['hypothesis']} (Alternatives: {h['alternative_explanations']})"
                for h in causal_hypotheses[:3]
            )
        else:
            synthesis_lines.append("  - [N/A] Nominal operational state observed. (Alternatives: None)")

        synthesis_lines.append("\n5. ACTIONABLE RECOMMENDATIONS & NDT VERIFICATION STEPS:")
        if recommendations:
            synthesis_lines.extend(
                f"  - [{rec['finding_uid']}] Action: {rec['corrective_action']} | Verification: {rec['verification_procedure']}"
                for rec in recommendations[:3]
            )
        else:
            synthesis_lines.append("  - [N/A] Action: Maintain routine inspection interval. | Verification: Standard shift visual check.")

        synthesis_text = "\n".join(synthesis_lines)

        return {
            "synthesis_text": synthesis_text,
            "observations": observations,
            "model_predictions": model_predictions,
            "retrieved_standards": retrieved_standards,
            "causal_hypotheses": causal_hypotheses,
            "recommendations": recommendations,
            "citation_count": len(retrieved_standards)
        }

evidence_synthesizer = EvidenceSynthesizer()
