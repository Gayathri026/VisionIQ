# VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing

**Gayathri R**$^1$, **Monica T S**$^1$, **Dhamodharan R**$^1$, and **Ms. Gayathri S**$^2$  
$^1$Department of Computer Science and Engineering, Dr. Mahalingam College of Engineering and Technology, Pollachi, Tamil Nadu, India  
$^2$Assistant Professor, Department of Computer Science and Engineering, Dr. Mahalingam College of Engineering and Technology, Pollachi, Tamil Nadu, India  

*Target Journal:* **IEEE Access** | *Course Code:* 23CSP701 (BE Project Phase II)  

---

## Abstract

Vision-language applications often execute a monolithic, fixed suite of computer-vision operations irrespective of the semantic requirements of a user's natural-language inquiry. Such static architectures introduce excessive computational overhead, increase end-to-end latency, and produce extraneous outputs that impair explainability. In this paper, we present **VisionIQ**, an adaptive, query-aware multi-agent vision intelligence framework that dynamically routes requests to twelve specialized vision and cognitive agents based on natural-language intent, estimated evidence capability, computational burden, and expected execution latency. The system features a conditional orchestration engine formulated in LangGraph, object detection via Ultralytics YOLOv11, optical character recognition via EasyOCR, morphometric surface defect localization, spatial scene understanding, evidence validation checking, retrieval-augmented generation (RAG) referencing authenticated OSHA and ASTM engineering standards, and downloadable PDF report generation with visual evidence provenance. We formalize a cost-aware routing function $S(a, q) = \alpha R(a,q) + \delta E(a) - \beta C(a) - \gamma L(a)$ calibrated with operating parameters ($\alpha=0.50, \beta=0.20, \gamma=0.15, \delta=0.15$, threshold $\tau=0.45$). Evaluated across a held-out benchmark suite of industrial image and video queries, VisionIQ achieves an agent-selection F1 score of **91.8%**, reduces median latency from **1,200.0 ms** to **555.0 ms** (a **53.8%** reduction), invokes a mean of **4.1** agents per query versus **9.0** in a fixed pipeline, and achieves a **100%** failure recovery rate under simulated agent faults.

**Index Terms** — VisionIQ, Computer Vision, Multi-Agent Systems, Dynamic Orchestration, Cost-Aware Routing, LangGraph, Retrieval-Augmented Generation, IEEE Access.

---

## I. Introduction

Computer-vision systems deployed in industrial, manufacturing, and construction domains are increasingly expected to interact through natural-language interfaces. A plant supervisor inspecting a manufacturing facility may ask diverse questions ranging from *"Are workers wearing hard hats and high-visibility vests?"* to *"What is the machine serial number on the rating plate?"* or *"Why did the hydraulic press halt suddenly during production?"*

Conventional vision-language systems commonly employ one of two extremes:
1. **Monolithic Multi-Modal Large Language Models (MLLMs)**: While expressive, large vision-language backbones incur high inference latency, require substantial GPU memory, and suffer from perceptual hallucinations when asked to inspect minute surface scratches or fine serial numbers.
2. **Fixed Static Vision Pipelines**: Standard industrial platforms run a hard-coded sequence of detectors, OCR engines, and segmentation algorithms on every uploaded image or video frame. This unconditional execution wastes compute, increases latency on simple queries, and clutters user dashboards with irrelevant findings.

To address these shortcomings, we present **VisionIQ**, a modular, cost-aware multi-agent vision intelligence framework. The primary research contributions of this paper are:
1. **Mathematical Cost-Aware Routing Formulation**: An explicit objective function balancing query relevance and evidence capability against computational cost and latency penalties, supported by deterministic fallback and dependency constraints.
2. **Specialized Multi-Agent Decomposition**: Independent agents for Object Detection (YOLOv11), PPE Compliance (with anatomical spatial association and explicit uncertainty preservation), OCR (EasyOCR), Industrial Surface Defect Inspection (NEU classes), Spatial Scene Understanding, Evidence Validation, and RAG Knowledge Retrieval.
3. **Temporal Video Tracking & Deduplication**: Frame sampling combined with IoU-based Multi-Object Tracking to prevent counting identical workers or machines multiple times across video streams.
4. **Empirical Comparative Evaluation & Ablation**: Experimental verification against fixed-pipeline and rule-based baselines measuring routing quality, perception metrics, latency distributions, and fault recovery.

---

## II. Related Work and Research Gap

Recent literature has explored agentic workflows for computer vision. Systems such as Visual ChatGPT and HuggingGPT coordinate specialized tools using LLM planning. However, existing multi-agent systems rely almost exclusively on unconstrained LLM zero-shot routing. They lack explicit mathematical cost penalties, do not model expected execution latency, and rarely preserve verifiable visual provenance. Furthermore, heuristic defect detection systems frequently conflate basic image gradient edges with actual material defects. VisionIQ resolves these gaps by uniting a calibrated routing formulation with graph-based fault-tolerant execution and authenticated standards grounding.

---

## III. System Architecture

The VisionIQ application architecture separates browser presentation, request validation, conditional workflow orchestration, and data persistence:

```
[React 18 + Vite + TypeScript UI]
       │ (HTTP REST / FormData / WebSockets)
[FastAPI Asynchronous Gateway]
       │
[Cost-Aware Dynamic Router] ── (Eq. 1 Scoring & Dependency Resolution)
       │
[LangGraph Conditional Workflow Engine]
 ├── Orchestrator Node (Intent Classification & Trace Init)
 ├── Vision Specialists Node (YOLOv11, PPE Assoc, Defect, OCR, Scene)
 ├── Evidence Validation Node (Cross-referencing & Contradiction Filter)
 ├── Cognitive Specialists Node (RAG Standards, RCA, Recommendations)
 └── Response Synthesis Node (Annotation Renderer, ReportLab PDF, SQLite)
```

---

## IV. Cost-Aware Dynamic Agent Selection Formulation

Let $q$ denote a natural-language query and $\mathcal{A} = \{a_1, a_2, \dots, a_M\}$ denote the universe of available specialist agents. For each candidate agent $a \in \mathcal{A}$, the planner estimates:
- $R(a, q) \in [0, 1]$: Query relevance score.
- $E(a) \in [0, 1]$: Expected evidence capability.
- $C(a) \in [0, 1]$: Normalized computational cost.
- $L(a) \in [0, 1]$: Normalized expected execution latency.

The routing score $S(a, q)$ is formulated as:
$$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a) \quad \text{(1)}$$

Subject to calibrated operating weights:
$$\alpha = 0.50, \quad \beta = 0.20, \quad \gamma = 0.15, \quad \delta = 0.15 \quad \text{(2)}$$

An agent is selected if its score meets the eligibility threshold $\tau = 0.45$:
$$\mathcal{A}_q = \{a \in \mathcal{A} : S(a, q) \ge \tau\} \quad \text{(3)}$$

### Dependency & Evidence Constraints
1. **RCA Dependency**: If $a_{\text{rca}} \in \mathcal{A}_q$, then $\{a_{\text{object}}, a_{\text{defect}}, a_{\text{ocr}}, a_{\text{ppe}}\} \cap \mathcal{A}_q \ne \emptyset$.
2. **PPE Association Dependency**: If $a_{\text{ppe}} \in \mathcal{A}_q$, then $a_{\text{object}} \in \mathcal{A}_q$ to provide anatomical reference bounds.
3. **Minimum Sufficient Evidence**: If $\mathcal{A}_q \cap \mathcal{A}_{\text{perception}} = \emptyset$, deterministic fallback automatically activates $\{a_{\text{object}}, a_{\text{scene}}\}$.

---

## V. Experimental Results & Discussion

### A. Comparative Benchmark Results

Table I summarizes empirical performance across a held-out benchmark query set comparing the three configurations under identical hardware and model parameters:

#### TABLE I: EMPIRICAL PERFORMANCE COMPARISON ACROSS SYSTEM CONFIGURATIONS

| Metric | Configuration A: Fixed Pipeline | Configuration B: Rule-Based Routing | Configuration C: Adaptive VisionIQ (Proposed) |
| :--- | :---: | :---: | :---: |
| **Test Query Count** | 10 | 10 | 10 |
| **Agent Selection Precision** | 0.521 | 0.742 | **0.933** |
| **Agent Selection Recall** | 1.000 | 0.783 | **0.917** |
| **Agent Selection F1 Score** | 0.672 | 0.751 | **0.918** |
| **Median Latency ($T_{\text{med}}$)** | 1,200.0 ms | 920.0 ms | **555.0 ms** |
| **95th Percentile Latency ($T_{\text{p95}}$)** | 1,380.0 ms | 1,175.0 ms | **685.0 ms** |
| **Latency Reduction ($\Delta T$)** | 0.0% | 23.3% | **53.8%** |
| **Mean Agents Invoked / Query** | 9.0 | 4.0 | **4.1** |
| **Failure Recovery Rate** | 80.0% | 90.0% | **100.0%** |
| **Object Detection mAP@0.50** | 0.884 | 0.884 | 0.884 |
| **OCR Word Error Rate (WER)** | 0.082 | 0.082 | 0.082 |

### B. Ablation Study

Table II evaluates the architectural contribution of individual components:

#### TABLE II: ABLATION STUDY RESULTS

| Ablation Variant | Routing F1 | Median Latency ($T_{\text{med}}$) | Mean Agents Invoked | Primary Architectural Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Full VisionIQ (Eq. 1)** | **91.8%** | **555.0 ms** | **4.1** | Optimal trade-off between evidence and efficiency. |
| **Ablation 1: No Cost Penalties ($\beta=0, \gamma=0$)** | 84.2% | 885.0 ms | 6.8 | Over-invokes expensive agents for simple text queries. |
| **Ablation 2: No RAG Grounding** | 83.8% | 510.0 ms | 3.6 | Disables authoritative OSHA/ASTM standard references. |

---

## VI. Threats to Validity and Limitations

1. **Threshold Sensitivity**: The operating threshold $\tau = 0.45$ was calibrated empirically; domain adaptation to specialized medical or aerospace imagery may require parameter re-tuning.
2. **2D Optical Cues vs Internal Physics**: Optical cameras capture surface manifestations only; root-cause analyses are explicitly reported as qualified hypotheses and require physical non-destructive testing (NDT) for conclusive verification.
3. **Lighting & Occlusion Constraints**: Dense occlusions can obscure PPE elements; VisionIQ treats occluded workers as *Uncertain* rather than asserting definite noncompliance.

---

## VII. Conclusion

VisionIQ demonstrates that dynamically routing visual questions to specialized, cost-aware agents achieves superior efficiency, reduced latency, and enhanced explainability compared to monolithic pipelines. By pairing LangGraph conditional orchestration with authentic RAG standards grounding and automated PDF reporting, VisionIQ provides a dependable foundation for industrial vision intelligence.

---

## Acknowledgment
The authors express their sincere gratitude to **Ms. Gayathri S**, Assistant Professor, Department of Computer Science and Engineering, Dr. Mahalingam College of Engineering and Technology, Pollachi, for her invaluable guidance and academic support throughout the course of this project.
