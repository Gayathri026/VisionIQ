# VisionIQ: Scientific Limitations, Threats to Validity, and Responsible-Use Report

**Authors:** Gayathri S, Gayathri R, Monica T S, Dhamodharan R  
**Affiliation:** Department of Computer Science and Engineering, Dr. Mahalingam College of Engineering and Technology, Anna University  
**Target Publication:** IEEE Access (Special Section on Multi-Agent Vision-Language Architectures)  
**Evaluation Target:** BE CSE Final Year Project (Course 23CSP701)

---

## 1. Executive Summary

This document provides a transparent, scientifically rigorous accounting of the assumptions, boundary conditions, empirical constraints, and operational limitations of the **VisionIQ** multi-agent vision intelligence framework. In strict accordance with IEEE Access research integrity standards and ACM ethical deployment principles, we explicitly distinguish validated empirical capabilities from current design approximations.

Automated computer vision systems deployed in industrial inspection and personnel safety monitoring operate in high-consequence domains. False negatives (e.g., missed personal protective equipment [PPE] infractions or undetected structural steel fissures) present direct life-safety and asset integrity risks, while false positives induce operational friction and alarm fatigue. This report documents known threats to validity and outlines mandatory human-in-the-loop protocols.

---

## 2. Threats to Validity

### 2.1 Internal Validity
1. **Routing Score Parameter Sensitivity:**
   The multi-criteria routing formulation in Equation (1):
   $$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a)$$
   utilizes configured engineering weights ($\alpha=0.40, \delta=0.30, \beta=0.15, \gamma=0.15$) and an activation threshold $\tau=0.25$. While these parameters achieve high routing F1 scores across our benchmark test cases, they represent calibrated engineering heuristics rather than globally optimal closed-form solutions. Variation in natural language phrasing across non-native English operators can introduce variance into relevance score $R(a, q)$.
2. **Deterministic Intent Fallback Reliance:**
   When language model intent inference is ambiguous or times out, the deterministic keyword-matching fallback provides conservative routing. However, highly idiosyncratic vocabulary or compound domain jargon absent from the keyword dictionary defaults to general object detection, potentially requiring user query refinement.
3. **Execution Latency Measurement Variance:**
   Reported latency metrics were measured on a dedicated standard CPU workstation (Intel x86_64, Windows Subsystem). While model warm-up passes were enforced prior to benchmark logging, operating system thread scheduling and background I/O introduce minor stochastic variance ($\pm 5\%$). All latency comparisons must be interpreted relative to the paired fixed-pipeline baseline run under identical hardware conditions.

### 2.2 Construct Validity
1. **Distinction Between Anomaly Cues and Certified NDT:**
   VisionIQ employs a dual-layer defect architecture:
   - *Layer 1 (Neural Classification):* A fine-tuned NEU DefectNet CNN classifying six standardized steel defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).
   - *Layer 2 (Heuristic Cues):* OpenCV-based edge gradients, Laplacian variance blur estimation, and contour density analysis.
   Heuristic cues serve strictly as localized exploratory signals. They **do not** constitute non-destructive testing (NDT) certification under ASTM E165 (Liquid Penetrant Examination) or ASTM E709 (Magnetic Particle Testing).
2. **PPE Anatomical Association Approximations:**
   The PPE Safety Specialist isolates worker bounding boxes and segments anatomical height zones (cranial zone: 0–25% vertical span; torso zone: 25–65% vertical span). While spatial containment and IoU heuristics reliably associate helmets and vests under upright poses, severe occlusion, extreme camera angles, or crowded workers overlapping along the line of sight can cause ambiguous entity attribution.
3. **Evidence Support Scoring Metric:**
   The automated Evidence Support Score $E_{\text{supp}} \in [0, 1]$ measures whether the set of executed agents satisfies the ground-truth required evidence modalities for the target query. While highly correlated with task completeness, it does not evaluate subjective prose aesthetics of downstream LLM synthesis.

### 2.3 External Validity
1. **Domain Generalization:**
   Perception models were trained on construction safety and steel manufacturing datasets. Generalization to novel industrial domains (e.g., semiconductor wafer fab inspection, aerospace composite delamination, food packaging) requires fine-tuning on domain-specific annotated datasets.
2. **Lighting and Sensor Degradation:**
   Sub-optimal environmental illumination (< 50 lux), lens distortion, motion blur, and industrial dust or steam can degrade detection confidence and OCR character recognition rates.

---

## 3. Failure Modes and Mitigation Strategies

| Failure Mode | Root Cause | Observable Symptom | VisionIQ Mitigation Mechanism |
| :--- | :--- | :--- | :--- |
| **Ambiguous Worker-PPE Attribution** | Multiple overlapping workers in line of sight | Helmet box overlaps two worker cranial zones | Stable Worker IDs (`W-01`, `W-02`); flagged with `human_review_required = True`. |
| **Heuristic Anomaly False Alarm** | Surface glare, grease stains, or shadows | Localized Laplacian variance spike | Defect classified as unconfirmed heuristic cue; mandatory NDT verification flagged. |
| **Specialist Execution Timeout** | Heavy model inference during transient spike | Agent execution exceeds per-node threshold | LangGraph fault isolation catches timeout; partial results synthesized with warning. |
| **RAG Passage Misalignment** | User query with novel phrasing | Lower similarity score (< 0.30) | Strict relevance thresholding drops irrelevant chunks; falls back to internal model knowledge. |
| **Path Traversal / Injection** | Malicious user input with `../` tokens | Attempted unauthorized file access | Strict regex validation (`^[a-zA-Z0-9_\-]+$`) rejects unsafe session IDs with HTTP 400. |

---

## 4. Responsible AI and Safety-Critical Deployment Guidelines

1. **Human-in-the-Loop Imperative:**
   VisionIQ is explicitly designed as an **augmented intelligence inspection assistant**, not an autonomous decision-maker. All critical safety infractions and structural defect classifications must be validated by certified on-site safety officers or licensed NDT inspectors prior to shutting down machinery or initiating punitive actions.
2. **Epistemological Provenance Separation:**
   Every generated output, user-facing finding card, and ReportLab PDF report enforces strict epistemological separation across five distinct layers:
   - *Direct Visual Observations:* Raw bounding boxes and sensor pixel coordinates.
   - *Model Predictions:* Confidence probabilities with explicit model versioning.
   - *Regulatory Standards:* Authoritative quoted excerpts with source URIs (OSHA, ASTM, ISO).
   - *Causal Inferences:* Root-cause hypotheses marked as tentative hypotheses.
   - *Recommendations:* Prescribed corrective and verification procedures.
3. **Privacy and Video Retention Protocols:**
   The video and live camera services process frames incrementally without persistent buffering of uncompressed video streams. Facial identities are not biometric enrollment vectors; personnel are tracked via temporary intra-session tracking IDs (`Track-01`, `Track-02`) that expire upon session termination.
4. **Data Protection and Audit Integrity:**
   All inspection findings, Equation (1) routing traces, and generated PDF reports are persisted with SHA-256 session integrity checksums to prevent post-hoc tampering in regulated compliance audits.

---

## 5. Conclusion and Roadmap

The limitations documented herein define the operational boundaries of VisionIQ v1.0. Ongoing research will focus on:
- Closed-form reinforcement learning calibration of routing weights ($\alpha, \beta, \gamma, \delta, \tau$).
- Temporal graph neural networks for multi-worker pose tracking under continuous heavy occlusion.
- Real-time FP16 TensorRT model quantization on embedded edge devices (NVIDIA Jetson AGX Orin).
