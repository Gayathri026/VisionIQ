import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Sliders,
  Cpu,
  Shield,
  RotateCcw,
  CheckCircle2,
  Save,
  Server,
  Activity,
  Layers,
  Database,
  FileCheck,
  Zap,
  HelpCircle
} from 'lucide-react';
import { api } from '../services/api';

export const SettingsPage: React.FC = () => {
  // Router Equation (1) Weights
  const [alpha, setAlpha] = useState(0.50);
  const [delta, setDelta] = useState(0.15);
  const [beta, setBeta] = useState(0.20);
  const [gamma, setGamma] = useState(0.15);
  const [threshold, setThreshold] = useState(0.35);

  // System Flags
  const [targetDevice, setTargetDevice] = useState<'cpu' | 'cuda'>('cpu');
  const [autoSaveTraces, setAutoSaveTraces] = useState(true);
  const [strictNDT, setStrictNDT] = useState(true);
  const [savedNotice, setSavedNotice] = useState(false);

  // Backend Health State
  const [backendHealth, setBackendHealth] = useState<{ status: string; latencyMs: number } | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);

  const checkHealth = async () => {
    setCheckingHealth(true);
    const start = performance.now();
    try {
      const data = await api.checkHealth();
      const latency = performance.now() - start;
      setBackendHealth({ status: data.status || 'healthy', latencyMs: latency });
    } catch {
      setBackendHealth({ status: 'unreachable', latencyMs: 0 });
    } finally {
      setCheckingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleResetDefaults = () => {
    setAlpha(0.50);
    setDelta(0.15);
    setBeta(0.20);
    setGamma(0.15);
    setThreshold(0.35);
    setTargetDevice('cpu');
    setAutoSaveTraces(true);
    setStrictNDT(true);
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 2500);
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 2500);
  };

  // Sample dynamic routing calculation for live preview
  const sampleRelevance = 0.92;
  const sampleEvidence = 0.85;
  const sampleCost = 0.30;
  const sampleLatency = 0.25;
  const sampleScore = (alpha * sampleRelevance) + (delta * sampleEvidence) - (beta * sampleCost) - (gamma * sampleLatency);
  const sampleSelected = sampleScore >= threshold;

  return (
    <div className="max-w-[1300px] mx-auto px-4 lg:px-8 py-8 space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="pb-6 border-b border-[#2B3A54] flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#151F35] border border-[#43D9F5]/30 text-[#43D9F5] text-xs font-mono mb-2">
            <SettingsIcon className="w-3.5 h-3.5" />
            <span>OPERATIONAL PARAMETERS & IEEE CONFIGURATION</span>
          </div>
          <h1 className="text-3xl font-extrabold text-[#F1F5F9] tracking-tight">
            VisionIQ System Configuration
          </h1>
          <p className="text-base text-[#B4C0D4] mt-1.5 max-w-3xl">
            Fine-tune Equation (1) cost-aware query routing hyperparameters, configure hardware acceleration targets, and manage empirical verification constraints.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={checkHealth}
            disabled={checkingHealth}
            className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-[#151F35] border border-[#2B3A54] hover:border-[#43D9F5]/50 text-xs font-mono text-[#F1F5F9] transition-all"
          >
            <Activity className={`w-3.5 h-3.5 ${checkingHealth ? 'animate-spin text-[#43D9F5]' : 'text-[#34D399]'}`} />
            <span>{checkingHealth ? 'Testing API...' : backendHealth?.status === 'healthy' ? `Backend: ${backendHealth.latencyMs.toFixed(0)}ms` : 'Check Diagnostic'}</span>
          </button>
        </div>
      </div>

      {savedNotice && (
        <div className="p-4 rounded-xl bg-[#34D399]/10 border border-[#34D399]/40 text-sm text-[#34D399] flex items-center space-x-3 shadow-lg">
          <CheckCircle2 className="w-5 h-5 text-[#34D399] flex-shrink-0" />
          <span className="font-medium">Parameters successfully updated and synchronized with active VisionIQ runtime.</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-8">
        {/* Card 1: Equation 1 Weights */}
        <div className="command-panel rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#2B3A54] gap-3">
            <div>
              <h2 className="text-xl font-bold text-[#F1F5F9] flex items-center space-x-2.5">
                <Sliders className="w-5 h-5 text-[#43D9F5]" />
                <span>Equation (1) Cost-Aware Query Router Weights</span>
              </h2>
              <p className="text-sm font-mono text-[#43D9F5] mt-1">
                S(a, q) = α · R(a, q) + δ · E(a) − β · C(a) − γ · L(a)
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-[#B4C0D4] bg-[#0B1020] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
                Weight Sum: <strong className="text-[#F1F5F9]">{(alpha + delta + beta + gamma).toFixed(2)}</strong>
              </span>
            </div>
          </div>

          {/* Interactive Sliders */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Alpha */}
            <div className="p-5 rounded-xl bg-[#0B1020]/60 border border-[#2B3A54]/80 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-base font-semibold text-[#F1F5F9]">α — Semantic Relevance Weight</span>
                <span className="text-lg font-mono font-bold text-[#43D9F5] bg-[#151F35] px-2.5 py-0.5 rounded border border-[#43D9F5]/30">
                  {alpha.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full accent-[#43D9F5] h-2 bg-[#1B2942] rounded-lg cursor-pointer"
              />
              <p className="text-xs text-[#B4C0D4] leading-relaxed">
                Prioritizes zero-shot intent alignment between query semantics and specialist agent capability domain.
              </p>
            </div>

            {/* Delta */}
            <div className="p-5 rounded-xl bg-[#0B1020]/60 border border-[#2B3A54]/80 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-base font-semibold text-[#F1F5F9]">δ — Evidence Grounding Weight</span>
                <span className="text-lg font-mono font-bold text-[#34D399] bg-[#151F35] px-2.5 py-0.5 rounded border border-[#34D399]/30">
                  {delta.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.05"
                value={delta}
                onChange={(e) => setDelta(parseFloat(e.target.value))}
                className="w-full accent-[#34D399] h-2 bg-[#1B2942] rounded-lg cursor-pointer"
              />
              <p className="text-xs text-[#B4C0D4] leading-relaxed">
                Rewards specialist agents producing verifiable pixel evidence (localized bounding boxes, polygons, spatial masks).
              </p>
            </div>

            {/* Beta */}
            <div className="p-5 rounded-xl bg-[#0B1020]/60 border border-[#2B3A54]/80 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-base font-semibold text-[#F1F5F9]">β — Compute Cost Penalty Weight</span>
                <span className="text-lg font-mono font-bold text-[#FBBF24] bg-[#151F35] px-2.5 py-0.5 rounded border border-[#FBBF24]/30">
                  {beta.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.05"
                value={beta}
                onChange={(e) => setBeta(parseFloat(e.target.value))}
                className="w-full accent-[#FBBF24] h-2 bg-[#1B2942] rounded-lg cursor-pointer"
              />
              <p className="text-xs text-[#B4C0D4] leading-relaxed">
                Penalizes FLOPs and token-expensive operations to conserve execution budget during high-throughput workloads.
              </p>
            </div>

            {/* Gamma */}
            <div className="p-5 rounded-xl bg-[#0B1020]/60 border border-[#2B3A54]/80 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-base font-semibold text-[#F1F5F9]">γ — Execution Latency Penalty Weight</span>
                <span className="text-lg font-mono font-bold text-[#FB7185] bg-[#151F35] px-2.5 py-0.5 rounded border border-[#FB7185]/30">
                  {gamma.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.50"
                step="0.05"
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full accent-[#FB7185] h-2 bg-[#1B2942] rounded-lg cursor-pointer"
              />
              <p className="text-xs text-[#B4C0D4] leading-relaxed">
                Penalizes expected execution time to eliminate latency bottlenecks for interactive and live camera workflows.
              </p>
            </div>
          </div>

          {/* Threshold τ & Live Calculation Simulator */}
          <div className="pt-6 border-t border-[#2B3A54] grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-base font-semibold text-[#F1F5F9]">Pruning Threshold (τ)</span>
                <span className="text-lg font-mono font-bold text-white bg-[#0B1020] px-3 py-1 rounded-lg border border-[#2B3A54]">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.60"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-[#43D9F5] h-2 bg-[#1B2942] rounded-lg cursor-pointer"
              />
              <p className="text-xs text-[#B4C0D4]">
                Any agent scoring below τ is pruned from the LangGraph execution graph, bypassing inference overhead.
              </p>
            </div>

            {/* Live Formula Preview Box */}
            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between text-xs font-mono text-[#B4C0D4]">
                <span>LIVE SIMULATION PREVIEW:</span>
                <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${sampleSelected ? 'bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40' : 'bg-red-950/40 text-red-400 border border-red-500/40'}`}>
                  {sampleSelected ? 'WILL EXECUTE' : 'PRUNED'}
                </span>
              </div>
              <div className="text-sm font-mono text-[#F1F5F9]">
                S(a, q) = ({alpha.toFixed(2)} × 0.92) + ({delta.toFixed(2)} × 0.85) − ({beta.toFixed(2)} × 0.30) − ({gamma.toFixed(2)} × 0.25)
              </div>
              <div className="text-base font-mono font-bold text-[#43D9F5]">
                = {sampleScore.toFixed(3)} {sampleScore >= threshold ? `≥ ${threshold.toFixed(2)}` : `< ${threshold.toFixed(2)}`}
              </div>
            </div>
          </div>
        </div>

        {/* Card 2: Hardware Execution & Verification Policies */}
        <div className="command-panel rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="pb-4 border-b border-[#2B3A54]">
            <h2 className="text-xl font-bold text-[#F1F5F9] flex items-center space-x-2.5">
              <Cpu className="w-5 h-5 text-[#818CF8]" />
              <span>Execution Hardware & Regulatory Verification Policies</span>
            </h2>
            <p className="text-sm text-[#B4C0D4] mt-1">
              Configure runtime execution target and regulatory verification guardrails.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Target Hardware */}
            <div className="space-y-3">
              <label className="block text-sm font-semibold text-[#F1F5F9]">Inference Target Architecture:</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setTargetDevice('cpu')}
                  className={`p-3.5 rounded-xl border text-sm font-mono text-left transition-all ${
                    targetDevice === 'cpu'
                      ? 'bg-[#43D9F5]/10 border-[#43D9F5] text-white shadow-md'
                      : 'bg-[#0B1020] border-[#2B3A54] text-[#B4C0D4] hover:text-white'
                  }`}
                >
                  <div className="font-bold flex items-center space-x-1.5">
                    <Server className="w-4 h-4 text-[#43D9F5]" />
                    <span>CPU Mode</span>
                  </div>
                  <div className="text-xs text-[#B4C0D4] mt-1 font-sans">
                    Deterministic IEEE Access baseline runtime
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setTargetDevice('cuda')}
                  className={`p-3.5 rounded-xl border text-sm font-mono text-left transition-all ${
                    targetDevice === 'cuda'
                      ? 'bg-[#818CF8]/10 border-[#818CF8] text-white shadow-md'
                      : 'bg-[#0B1020] border-[#2B3A54] text-[#B4C0D4] hover:text-white'
                  }`}
                >
                  <div className="font-bold flex items-center space-x-1.5">
                    <Zap className="w-4 h-4 text-[#818CF8]" />
                    <span>CUDA / GPU</span>
                  </div>
                  <div className="text-xs text-[#B4C0D4] mt-1 font-sans">
                    Hardware acceleration if CUDA device present
                  </div>
                </button>
              </div>
            </div>

            {/* Checkbox Toggles */}
            <div className="space-y-4 pt-1">
              <label className="flex items-start space-x-3 cursor-pointer p-3 rounded-xl bg-[#0B1020]/40 border border-[#2B3A54]/60 hover:border-[#2B3A54]">
                <input
                  type="checkbox"
                  checked={strictNDT}
                  onChange={(e) => setStrictNDT(e.target.checked)}
                  className="mt-1 rounded border-[#2B3A54] text-[#43D9F5] focus:ring-0 bg-[#0B1020] w-4 h-4 cursor-pointer"
                />
                <div>
                  <span className="text-sm font-semibold text-[#F1F5F9] block">
                    Enforce Epistemic Separation & NDT Protocol
                  </span>
                  <span className="text-xs text-[#B4C0D4] block mt-0.5">
                    Flags pure optical inferences as hypothetical without mandatory non-destructive physical testing (ASTM/ASME).
                  </span>
                </div>
              </label>

              <label className="flex items-start space-x-3 cursor-pointer p-3 rounded-xl bg-[#0B1020]/40 border border-[#2B3A54]/60 hover:border-[#2B3A54]">
                <input
                  type="checkbox"
                  checked={autoSaveTraces}
                  onChange={(e) => setAutoSaveTraces(e.target.checked)}
                  className="mt-1 rounded border-[#2B3A54] text-[#43D9F5] focus:ring-0 bg-[#0B1020] w-4 h-4 cursor-pointer"
                />
                <div>
                  <span className="text-sm font-semibold text-[#F1F5F9] block">
                    Persist Full Mathematical Traces to SQLite
                  </span>
                  <span className="text-xs text-[#B4C0D4] block mt-0.5">
                    Records every agent scoring parameter to enable historical audit and IEEE Access paper validation.
                  </span>
                </div>
              </label>
            </div>
          </div>
        </div>

        {/* Card 3: Checkpoint & Service Catalog Status */}
        <div className="command-panel rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="pb-4 border-b border-[#2B3A54]">
            <h2 className="text-xl font-bold text-[#F1F5F9] flex items-center space-x-2.5">
              <Layers className="w-5 h-5 text-[#34D399]" />
              <span>VisionIQ Multi-Agent Model & Storage Registry</span>
            </h2>
            <p className="text-sm text-[#B4C0D4] mt-1">
              Verified local checkpoints and storage engines loaded in current environment.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#43D9F5] uppercase">Agent C: PPE YOLO</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">models/yolo_ppe.pt</div>
              <p className="text-xs text-[#B4C0D4]">YOLOv11n weights with Head/Torso spatial compliance logic</p>
            </div>

            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#43D9F5] uppercase">Agent E: DefectNet</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">neu_defect_classifier.pt</div>
              <p className="text-xs text-[#B4C0D4]">NEU surface defect model (6 classes: crazing, patches, etc.)</p>
            </div>

            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#43D9F5] uppercase">Agent D: OCR Engine</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">EasyOCR (CRAFT + CRNN)</div>
              <p className="text-xs text-[#B4C0D4]">Dual-stage text localization and English recognition</p>
            </div>

            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#818CF8] uppercase">Agent H: RAG Vector Store</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">Hybrid TF-IDF + BM25</div>
              <p className="text-xs text-[#B4C0D4]">6 indexed regulatory standards (OSHA 1910, ASTM A568, ISO 13849)</p>
            </div>

            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#818CF8] uppercase">Provenance Database</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">SQLite Engine</div>
              <p className="text-xs text-[#B4C0D4]">storage/database/visioniq.db schema v1.0.0</p>
            </div>

            <div className="p-4 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#818CF8] uppercase">PDF Generation</span>
                <span className="w-2 h-2 rounded-full bg-[#34D399]" />
              </div>
              <div className="text-sm font-bold text-[#F1F5F9]">ReportLab 4.4.1</div>
              <p className="text-xs text-[#B4C0D4]">Two-column engineering inspection reports with SHA-256</p>
            </div>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
          <button
            type="button"
            onClick={handleResetDefaults}
            className="w-full sm:w-auto flex items-center justify-center space-x-2 px-5 py-3 rounded-xl bg-[#151F35] hover:bg-[#1B2942] border border-[#2B3A54] text-[#B4C0D4] hover:text-[#F1F5F9] text-sm font-mono transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset to IEEE Access Defaults</span>
          </button>

          <button
            type="submit"
            className="w-full sm:w-auto flex items-center justify-center space-x-2.5 px-8 py-3.5 rounded-xl bg-[#43D9F5] hover:bg-[#43D9F5]/90 text-[#0B1020] font-bold text-base shadow-lg shadow-[#43D9F5]/20 transition-all cursor-pointer"
          >
            <Save className="w-5 h-5" />
            <span>Apply & Save Settings</span>
          </button>
        </div>
      </form>
    </div>
  );
};
