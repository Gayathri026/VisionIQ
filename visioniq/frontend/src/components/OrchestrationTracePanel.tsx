import React, { useState } from 'react';
import { AgentTrace } from '../types';
import { Calculator, ChevronDown, ChevronUp, Clock, CheckCircle2, XCircle, AlertCircle, Play, Info, AlertTriangle, HelpCircle } from 'lucide-react';
import { api } from '../services/api';

interface Props {
  traces: AgentTrace[];
  detectedIntent?: string;
  totalLatencyMs?: number;
  onPlanGenerated?: (plan: any) => void;
}

const AGENT_CATALOG: Record<string, { purpose: string; model: string; evidenceType: string }> = {
  agent_a_intent: { purpose: "Zero-Shot Query Intent Classification", model: "DistilBERT / Fallback Rules", evidenceType: "Intent Class & Confidence" },
  agent_b_object: { purpose: "General Object Detection & Bounding Box Localization", model: "YOLOv11n (COCO)", evidenceType: "XYXY Bounding Boxes" },
  agent_c_ppe: { purpose: "Worker Safety Compliance & Head/Torso Anatomical Zoning", model: "VisionIQ-PPE-YOLO11n (Roboflow)", evidenceType: "Person-PPE Spatial Tuples" },
  agent_d_ocr: { purpose: "Serial Number & Machinery Label Text Region Localization", model: "EasyOCR (CRAFT + CRNN)", evidenceType: "Text Polygons & Strings" },
  agent_e_defect: { purpose: "Industrial Steel Surface Defect Classification", model: "IndustrialDefectNet (NEU-DET)", evidenceType: "Softmax Defect Class Probabilities" },
  agent_f_scene: { purpose: "Continuous Spatial Anomaly Localization", model: "PatchCore / MVTec-AD", evidenceType: "Anomaly Heatmap & Peak Score" },
  agent_g_evidence: { purpose: "Multi-Source Observation & Finding Provenance Aggregator", model: "Rule-Based Aggregator", evidenceType: "Epistemological Layers" },
  agent_h_rag: { purpose: "Regulatory Standards Retrieval (OSHA, ASTM, ISO)", model: "Hybrid TF-IDF + BM25 Vector Store", evidenceType: "Authenticated Standard Passages" },
  agent_i_rca: { purpose: "Hypothetical Root Cause & Epistemic Uncertainty Estimation", model: "Evidence Synthesizer", evidenceType: "Causal Hypotheses & NDT Steps" },
  agent_j_recommendation: { purpose: "Non-Destructive Testing (NDT) & Corrective Action Engine", model: "ASTM / ISO Verification Map", evidenceType: "Mandatory Corrective Action" },
};

export const OrchestrationTracePanel: React.FC<Props> = ({ traces, detectedIntent, totalLatencyMs, onPlanGenerated }) => {
  const [expanded, setExpanded] = useState(true);
  const [selectedAgentDetails, setSelectedAgentDetails] = useState<string | null>(null);
  const [simulationQuery, setSimulationQuery] = useState('');
  const [simulating, setSimulating] = useState(false);
  const [simulatedPlan, setSimulatedPlan] = useState<any | null>(null);

  const activeTraces = simulatedPlan ? Object.values(simulatedPlan.traces) as AgentTrace[] : traces;
  const currentIntent = simulatedPlan ? simulatedPlan.detected_intent : detectedIntent;

  const testPresets = [
    { label: "A. Read Serial", query: "Read the serial number on this machine." },
    { label: "B. Count People", query: "Count all visible people." },
    { label: "C. Check Helmets", query: "Check whether workers are wearing helmets." },
    { label: "D. Locate Cracks", query: "Locate cracks or scratches on this metal surface." },
    { label: "E. Scene & Damage", query: "Describe the scene and explain the possible damage." }
  ];

  const handleSimulatePlan = async (queryText: string) => {
    setSimulationQuery(queryText);
    setSimulating(true);
    try {
      const plan = await api.getOrchestrationPlan(queryText);
      setSimulatedPlan(plan);
      if (onPlanGenerated) {
        onPlanGenerated(plan);
      }
    } catch (err) {
      console.error("Simulation failed:", err);
    } finally {
      setSimulating(false);
    }
  };

  const getStatusBadge = (trace: AgentTrace) => {
    if (trace.error_message) {
      return (
        <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#FB7185]/20 text-[#FB7185] border border-[#FB7185]/40">
          <AlertTriangle className="w-3 h-3" />
          <span>FAILED</span>
        </span>
      );
    }

    if (trace.decision === 'SELECTED') {
      if (trace.execution_status === 'PENDING') {
        return (
          <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/40">
            <Clock className="w-3 h-3 animate-spin" />
            <span>RUNNING</span>
          </span>
        );
      }
      return (
        <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40">
          <CheckCircle2 className="w-3 h-3" />
          <span>COMPLETED</span>
        </span>
      );
    }

    if (trace.decision === 'MANDATORY_DEPENDENCY') {
      return (
        <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#43D9F5]/20 text-[#43D9F5] border border-[#43D9F5]/40">
          <Info className="w-3 h-3" />
          <span>DEPENDENCY</span>
        </span>
      );
    }

    if (trace.decision === 'FALLBACK') {
      return (
        <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/40">
          <AlertCircle className="w-3 h-3" />
          <span>FALLBACK</span>
        </span>
      );
    }

    return (
      <span className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-[#1B2942] text-[#64748B] border border-[#2B3A54]">
        <XCircle className="w-3 h-3" />
        <span>PRUNED</span>
      </span>
    );
  };

  return (
    <div className="command-panel rounded-2xl overflow-hidden mb-6">
      {/* Header */}
      <div
        className="px-6 py-4 flex items-center justify-between cursor-pointer border-b border-[#2B3A54] bg-[#10182B] hover:bg-[#151F35] transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-[#43D9F5]/20 text-[#43D9F5] border border-[#43D9F5]/30">
            <Calculator className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-[#F1F5F9]">Cost-Aware Dynamic Orchestration Trace</h3>
            <p className="text-xs text-[#B4C0D4] font-mono mt-0.5">
              Equation (1): S(a,q) = αR + δE − βC − γL | Detected Intent: <span className="text-[#43D9F5] font-bold">{currentIntent || 'General Inspection'}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {totalLatencyMs !== undefined && (
            <div className="flex items-center space-x-1.5 text-xs font-mono text-[#F1F5F9] bg-[#0B1020] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
              <Clock className="w-3.5 h-3.5 text-[#43D9F5]" />
              <span>{totalLatencyMs.toFixed(1)} ms</span>
            </div>
          )}
          {expanded ? <ChevronUp className="w-5 h-5 text-[#B4C0D4]" /> : <ChevronDown className="w-5 h-5 text-[#B4C0D4]" />}
        </div>
      </div>

      {/* Body */}
      {expanded && (
        <div className="p-6 overflow-x-auto space-y-5">
          {/* Quick Query Simulation Presets */}
          <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-[#F1F5F9] font-bold flex items-center gap-2">
                <Play className="w-4 h-4 text-[#43D9F5]" />
                <span>Dry-Run Routing Simulator (Test Benchmark Intents):</span>
              </span>
              {simulatedPlan && (
                <button
                  onClick={() => setSimulatedPlan(null)}
                  className="text-xs text-[#FBBF24] hover:underline font-mono cursor-pointer"
                >
                  Reset to Actual Execution Trace
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {testPresets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSimulatePlan(preset.query)}
                  disabled={simulating}
                  className="px-3 py-1.5 rounded-lg text-xs font-mono bg-[#151F35] hover:bg-[#1B2942] text-[#F1F5F9] border border-[#2B3A54] hover:border-[#43D9F5]/60 transition-all cursor-pointer"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Equation Legend with Active Parameters */}
          <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] text-xs font-mono grid grid-cols-2 sm:grid-cols-5 gap-3 text-[#B4C0D4]">
            <div><span className="text-[#43D9F5] font-bold">α (Relevance):</span> 0.50</div>
            <div><span className="text-[#34D399] font-bold">δ (Evidence):</span> 0.15</div>
            <div><span className="text-[#FBBF24] font-bold">β (Cost Penalty):</span> 0.20</div>
            <div><span className="text-[#FB7185] font-bold">γ (Latency Penalty):</span> 0.15</div>
            <div><span className="text-white font-bold">Pruning Threshold τ:</span> 0.35</div>
          </div>

          {/* Trace Execution Table */}
          <table className="w-full text-left text-sm font-mono">
            <thead>
              <tr className="border-b border-[#2B3A54] text-[#B4C0D4] text-xs uppercase tracking-wider">
                <th className="pb-3 font-semibold">Specialist Agent</th>
                <th className="pb-3 font-semibold text-center">R(a,q)</th>
                <th className="pb-3 font-semibold text-center">E(a)</th>
                <th className="pb-3 font-semibold text-center">C(a)</th>
                <th className="pb-3 font-semibold text-center">L(a)</th>
                <th className="pb-3 font-semibold text-center text-[#F1F5F9]">Score S(a,q)</th>
                <th className="pb-3 font-semibold text-center">Status</th>
                <th className="pb-3 font-semibold text-right">Runtime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#2B3A54]/50">
              {activeTraces.map((trace) => {
                const isSelected = trace.decision !== 'SKIPPED';
                const isExpanded = selectedAgentDetails === trace.agent_name;
                const meta = AGENT_CATALOG[trace.agent_name] || {
                  purpose: "Specialist Vision Node",
                  model: "Deterministic Module",
                  evidenceType: "Visual Inferences"
                };

                const alphaR = (0.50 * trace.relevance_score).toFixed(3);
                const deltaE = (0.15 * trace.evidence_capability).toFixed(3);
                const betaC = (0.20 * trace.normalized_cost).toFixed(3);
                const gammaL = (0.15 * trace.expected_latency).toFixed(3);

                return (
                  <React.Fragment key={trace.agent_name}>
                    <tr
                      onClick={() => setSelectedAgentDetails(isExpanded ? null : trace.agent_name)}
                      className={`hover:bg-[#1B2942]/60 cursor-pointer transition-colors ${
                        isSelected ? 'bg-[#151F35]/40' : 'opacity-50'
                      }`}
                    >
                      <td className="py-3.5 font-semibold text-[#F1F5F9] capitalize flex items-center space-x-2.5">
                        <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-[#43D9F5]' : 'bg-[#64748B]'}`} />
                        <span>{trace.agent_name.replace(/_/g, ' ')}</span>
                        <Info className="w-3.5 h-3.5 text-[#64748B]" />
                      </td>
                      <td className="py-3.5 text-center text-[#43D9F5]">{trace.relevance_score.toFixed(2)}</td>
                      <td className="py-3.5 text-center text-[#34D399]">{trace.evidence_capability.toFixed(2)}</td>
                      <td className="py-3.5 text-center text-[#FBBF24]">{trace.normalized_cost.toFixed(2)}</td>
                      <td className="py-3.5 text-center text-[#FB7185]">{trace.expected_latency.toFixed(2)}</td>
                      <td className="py-3.5 text-center font-bold text-[#F1F5F9]">
                        {trace.final_routing_score.toFixed(3)}
                      </td>
                      <td className="py-3.5 text-center">{getStatusBadge(trace)}</td>
                      <td className="py-3.5 text-right text-[#B4C0D4]">
                        {trace.actual_latency_ms > 0 ? `${trace.actual_latency_ms.toFixed(1)} ms` : '—'}
                      </td>
                    </tr>

                    {/* Expandable Technical Calculation & Agent Provenance */}
                    {isExpanded && (
                      <tr className="bg-[#0B1020] border-b border-[#2B3A54]">
                        <td colSpan={8} className="p-5 text-xs font-mono text-[#F1F5F9] border-l-4 border-[#43D9F5] space-y-3">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pb-3 border-b border-[#2B3A54]">
                            <div>
                              <span className="text-[#B4C0D4] block text-xs uppercase tracking-wider">Functional Role:</span>
                              <span className="text-[#F1F5F9] font-sans font-medium text-sm mt-0.5 block">{meta.purpose}</span>
                            </div>
                            <div>
                              <span className="text-[#B4C0D4] block text-xs uppercase tracking-wider">Registered Model:</span>
                              <span className="text-[#43D9F5] font-semibold text-sm mt-0.5 block">{meta.model}</span>
                            </div>
                            <div>
                              <span className="text-[#B4C0D4] block text-xs uppercase tracking-wider">Output Evidence Type:</span>
                              <span className="text-[#34D399] font-semibold text-sm mt-0.5 block">{meta.evidenceType}</span>
                            </div>
                          </div>

                          <div className="p-3.5 rounded-xl bg-[#151F35] border border-[#2B3A54] space-y-1.5">
                            <span className="text-[#43D9F5] font-bold text-xs uppercase tracking-wider">Mathematical Derivation:</span>
                            <div className="text-sm text-[#B4C0D4]">
                              S(a, q) = (0.50 × {trace.relevance_score.toFixed(2)}) + (0.15 × {trace.evidence_capability.toFixed(2)}) − (0.20 × {trace.normalized_cost.toFixed(2)}) − (0.15 × {trace.expected_latency.toFixed(2)})
                            </div>
                            <div className="text-sm text-[#F1F5F9] font-bold">
                              S(a, q) = {alphaR} + {deltaE} − {betaC} − {gammaL} = <span className="text-[#43D9F5]">{trace.final_routing_score.toFixed(3)}</span>
                              <span className="text-[#B4C0D4] font-normal ml-2">
                                ({trace.final_routing_score >= 0.35 ? '≥ Threshold 0.35 → SELECTED FOR EXECUTION' : '< Threshold 0.35 → PRUNED'})
                              </span>
                            </div>
                          </div>

                          <div className="text-xs text-[#B4C0D4]">
                            <strong className="text-[#F1F5F9]">Selection Rationale:</strong> {trace.selection_reason || "Calculated dynamically by Equation 1 cost-aware router."}
                          </div>

                          {trace.error_message && (
                            <div className="mt-2 p-3 rounded-xl bg-red-950/60 border border-red-500/40 text-red-300 text-xs">
                              <span className="font-bold">Error Isolated:</span> {trace.error_message}
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
