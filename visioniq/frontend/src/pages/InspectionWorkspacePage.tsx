import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import { AnalysisResponseData, HistorySession } from '../types';
import { EvidenceWorkspace } from '../components/EvidenceWorkspace';
import { OrchestrationTracePanel } from '../components/OrchestrationTracePanel';
import { FindingCards } from '../components/FindingCards';
import { RAGPanel } from '../components/RAGPanel';
import {
  Upload,
  Sparkles,
  AlertCircle,
  Loader2,
  Copy,
  Check,
  FileDown,
  History as HistoryIcon,
  Sliders,
  RefreshCw,
  SlidersHorizontal,
  ChevronRight,
  Crosshair,
  ShieldAlert,
  Clock,
  Layers,
  BookOpen,
  FileText
} from 'lucide-react';

const SAMPLE_QUERIES = [
  "Check whether workers are wearing safety helmets and reflective vests.",
  "Locate cracks, inclusions, or surface scratches on this metal plate.",
  "Read the serial number and machine rating on this industrial label.",
  "Describe the scene and explain the possible machinery damage causes."
];

export const InspectionWorkspacePage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const urlSessionId = searchParams.get('session');

  // Input & Query State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileMeta, setFileMeta] = useState<{ size: string; name: string } | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.45);
  const [routingThreshold, setRoutingThreshold] = useState<number>(0.35);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // Tab State in Right Intelligence Panel
  const [activeTab, setActiveTab] = useState<'findings' | 'trace' | 'rag' | 'synthesis'>('findings');

  // Execution & Results State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponseData | null>(null);
  const [selectedFindingUid, setSelectedFindingUid] = useState<string | null>(null);
  const [copiedSessionId, setCopiedSessionId] = useState(false);

  // Recent Sessions History Quick-Picker
  const [recentSessions, setRecentSessions] = useState<HistorySession[]>([]);
  const [showHistoryDrawer, setShowHistoryDrawer] = useState(false);

  // Load recent history sessions on mount
  useEffect(() => {
    const fetchRecent = async () => {
      try {
        const data = await api.getHistory(8);
        setRecentSessions(data);
      } catch (err) {
        console.error("Failed to load recent sessions", err);
      }
    };
    fetchRecent();
  }, []);

  // If session ID is provided via URL parameter, load it immediately
  useEffect(() => {
    if (urlSessionId) {
      loadSessionById(urlSessionId);
    }
  }, [urlSessionId]);

  const loadSessionById = async (sid: string) => {
    try {
      setLoading(true);
      setError(null);
      const session = await api.getSessionDetails(sid);
      setAnalysisResult(session);
      setQuery(session.user_query);
      if (session.input_url) {
        setPreviewUrl(session.input_url);
      }
      if (session.findings && session.findings.length > 0) {
        setSelectedFindingUid(session.findings[0].finding_uid);
      }
    } catch (err: any) {
      console.error(err);
      setError(`Failed to load session ${sid}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setFileMeta({
        name: file.name,
        size: file.size > 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : `${(file.size / 1024).toFixed(1)} KB`
      });
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleCopyId = () => {
    if (analysisResult?.session_id) {
      navigator.clipboard.writeText(analysisResult.session_id);
      setCopiedSessionId(true);
      setTimeout(() => setCopiedSessionId(false), 2000);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile && !analysisResult) {
      setError("Please select or drop an industrial image first.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter an inspection query.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let data: AnalysisResponseData;
      if (selectedFile) {
        data = await api.analyzeImage(selectedFile, query);
      } else {
        data = await api.getSessionDetails(analysisResult!.session_id);
      }
      setAnalysisResult(data);
      if (data.findings && data.findings.length > 0) {
        setSelectedFindingUid(data.findings[0].finding_uid);
      }
      // Refresh recent sessions list
      const updatedHistory = await api.getHistory(8);
      setRecentSessions(updatedHistory);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Analysis request failed. Please check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-[1800px] mx-auto px-4 lg:px-8 py-5 space-y-5">
      {/* 1. Command Center Top Header: Session Status & Actions */}
      <div className="command-panel p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5 flex-wrap">
          <div className="w-10 h-10 rounded-xl bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#43D9F5]">
            <Crosshair className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              Visual Inspection Command Center
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Multi-Agent Perception • Equation 1 Cost-Aware Query Routing • Verifiable Standards
            </p>
          </div>
        </div>

        {/* Session Stats & Quick Actions */}
        <div className="flex items-center space-x-3 flex-wrap">
          {analysisResult && (
            <>
              <div className="flex items-center space-x-2 bg-[#0B1020] px-3 py-1.5 rounded-lg border border-[#2B3A54] text-xs font-mono">
                <span className="text-[#8998AF]">UID:</span>
                <span className="text-[#43D9F5] font-bold">{analysisResult.session_id.slice(0, 12)}</span>
                <button
                  onClick={handleCopyId}
                  className="text-[#8998AF] hover:text-white p-0.5 rounded transition-colors"
                  title="Copy Full Session UID"
                >
                  {copiedSessionId ? <Check className="w-3.5 h-3.5 text-[#34D399]" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>

              <div className="flex items-center space-x-1.5 bg-[#0B1020] px-3 py-1.5 rounded-lg border border-[#2B3A54] text-xs font-mono">
                <span className="text-[#8998AF]">Intent:</span>
                <strong className="text-white">{analysisResult.detected_intent || 'General'}</strong>
              </div>

              <div className="flex items-center space-x-1.5 bg-[#0B1020] px-3 py-1.5 rounded-lg border border-[#2B3A54] text-xs font-mono">
                <Clock className="w-3.5 h-3.5 text-[#43D9F5]" />
                <span className="text-[#43D9F5] font-bold">{analysisResult.total_latency_ms.toFixed(1)} ms</span>
              </div>

              <a
                href={api.getPdfReportUrl(analysisResult.session_id)}
                target="_blank"
                rel="noreferrer"
                className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-[#43D9F5] hover:bg-[#38bdf8] text-[#0B1020] font-bold text-xs transition-all shadow-md shadow-[#43D9F5]/20"
              >
                <FileDown className="w-4 h-4" />
                <span>Export PDF Report</span>
              </a>
            </>
          )}

          <button
            onClick={() => setShowHistoryDrawer(!showHistoryDrawer)}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-[#1B2942] hover:bg-[#223252] border border-[#2B3A54] text-[#F1F5F9] text-xs font-medium transition-colors"
          >
            <HistoryIcon className="w-4 h-4 text-[#43D9F5]" />
            <span>Recent ({recentSessions.length})</span>
          </button>
        </div>
      </div>

      {/* Recent Sessions Drawer / Popdown */}
      {showHistoryDrawer && (
        <div className="command-panel p-4 bg-[#10182B] border-[#43D9F5]/40 space-y-2">
          <div className="flex items-center justify-between pb-2 border-b border-[#2B3A54]">
            <span className="text-xs font-mono uppercase text-[#43D9F5] font-bold">Recent Completed Analyses</span>
            <button onClick={() => setShowHistoryDrawer(false)} className="text-xs text-[#8998AF] hover:text-white">✕ Close</button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
            {recentSessions.map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  loadSessionById(s.id);
                  setShowHistoryDrawer(false);
                }}
                className="text-left p-2.5 rounded-lg bg-[#151F35] hover:bg-[#1B2942] border border-[#2B3A54] hover:border-[#43D9F5] transition-all"
              >
                <div className="text-xs font-semibold text-[#F1F5F9] truncate">{s.user_query}</div>
                <div className="text-[11px] font-mono text-[#8998AF] mt-1 flex items-center justify-between">
                  <span>{s.detected_intent}</span>
                  <span className="text-[#43D9F5] font-bold">{s.total_latency_ms ? `${s.total_latency_ms.toFixed(0)}ms` : ''}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 2. Main Two-Column Layout (Image Sized to Content, Independent of Right Side Length) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: Image Canvas & Query Console Sized to Image (self-start + sticky) */}
        <div className="lg:col-span-7 xl:col-span-8 space-y-4 self-start lg:sticky lg:top-20">
          {/* Main Visual Evidence Workspace (Full Focus) */}
          <div className="w-full">
            <EvidenceWorkspace
              originalUrl={previewUrl || analysisResult?.input_url}
              annotatedUrl={analysisResult?.annotated_url}
              pdfUrl={analysisResult ? api.getPdfReportUrl(analysisResult.session_id) : undefined}
              findings={analysisResult?.findings || []}
              selectedFindingUid={selectedFindingUid}
              onSelectFinding={setSelectedFindingUid}
            />
          </div>

          {/* Compact Query & Execution Console (Right beneath image) */}
          <div className="command-panel p-4 space-y-3.5">
            {/* Input Bar Row */}
            <div className="flex flex-col sm:flex-row items-stretch gap-3">
              {/* Image Upload Trigger */}
              <div className="flex-shrink-0">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="workspace-image-input"
                />
                <label
                  htmlFor="workspace-image-input"
                  className="h-full px-4 py-2.5 rounded-lg border border-[#2B3A54] hover:border-[#43D9F5] bg-[#10182B] hover:bg-[#1B2942] cursor-pointer flex items-center justify-center space-x-2 text-xs font-semibold text-[#F1F5F9] transition-all shadow-sm"
                >
                  <Upload className="w-4 h-4 text-[#43D9F5]" />
                  <span>{selectedFile ? 'Replace Image' : 'Load Image'}</span>
                </label>
              </div>

              {/* Natural Language Query Textarea / Input */}
              <div className="flex-1">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter inspection query (e.g. Check worker hardhats, detect cracks, read serial plate)..."
                  className="w-full h-full px-4 py-2.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-sm text-[#F1F5F9] placeholder-[#8998AF] focus:outline-none focus:border-[#43D9F5] font-sans"
                />
              </div>

              {/* Run Analysis Action Button */}
              <button
                onClick={handleRunAnalysis}
                disabled={loading}
                className="px-6 py-2.5 rounded-lg bg-gradient-to-r from-[#0284C7] to-[#43D9F5] hover:opacity-90 disabled:opacity-50 text-[#0B1020] font-extrabold text-sm shadow-lg shadow-[#43D9F5]/20 flex items-center justify-center space-x-2 transition-all flex-shrink-0"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-[#0B1020]" />
                    <span>Orchestrating...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-[#0B1020]" />
                    <span>Run Analysis</span>
                  </>
                )}
              </button>
            </div>

            {/* Error Message Alert */}
            {error && (
              <div className="p-3 rounded-lg bg-[#FB7185]/15 border border-[#FB7185]/40 text-xs text-[#FB7185] flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Sample Benchmark Query Quick Chips */}
            <div className="flex items-center space-x-2 overflow-x-auto pt-1">
              <span className="text-[11px] font-mono text-[#8998AF] uppercase tracking-wider font-semibold whitespace-nowrap">
                Sample Queries:
              </span>
              {SAMPLE_QUERIES.map((sq, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sq)}
                  className="px-2.5 py-1 rounded-md bg-[#10182B] hover:bg-[#1B2942] border border-[#2B3A54] hover:border-[#43D9F5] text-xs text-[#B4C0D4] hover:text-white whitespace-nowrap transition-colors"
                >
                  {sq.slice(0, 42)}...
                </button>
              ))}
            </div>

            {/* Optional Collapsible Settings: Routing Thresholds */}
            <div className="pt-2 border-t border-[#2B3A54]/60 flex items-center justify-between text-xs font-mono text-[#8998AF]">
              <div className="flex items-center space-x-4">
                <span>Confidence Cutoff: <strong className="text-white">{confidenceThreshold.toFixed(2)}</strong></span>
                <span>Selection Threshold: <strong className="text-[#43D9F5]">{routingThreshold.toFixed(2)}</strong></span>
              </div>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="text-[#43D9F5] hover:underline flex items-center space-x-1"
              >
                <Sliders className="w-3 h-3" />
                <span>{showSettings ? 'Hide Parameters' : 'Tune Eq. 1 Parameters'}</span>
              </button>
            </div>

            {showSettings && (
              <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
                <div>
                  <div className="flex justify-between mb-1">
                    <span>Detection Threshold (τ_c):</span>
                    <span className="text-white font-bold">{confidenceThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#43D9F5] h-1.5 bg-[#2B3A54] rounded-lg cursor-pointer"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span>Agent Selection Gating (τ_s):</span>
                    <span className="text-[#43D9F5] font-bold">{routingThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.20"
                    max="0.60"
                    step="0.05"
                    value={routingThreshold}
                    onChange={(e) => setRoutingThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#43D9F5] h-1.5 bg-[#2B3A54] rounded-lg cursor-pointer"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT COLUMN: Intelligence, Findings & Trace Panel (~35% width) */}
        <div className="lg:col-span-5 xl:col-span-4 space-y-4">
          {/* Tab Navigation Controls */}
          <div className="command-panel p-1.5 flex items-center justify-between text-xs font-mono">
            <button
              onClick={() => setActiveTab('findings')}
              className={`flex-1 py-2 px-2.5 rounded-lg font-bold transition-all text-center ${
                activeTab === 'findings'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Findings ({analysisResult?.findings?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('trace')}
              className={`flex-1 py-2 px-2.5 rounded-lg font-bold transition-all text-center ${
                activeTab === 'trace'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Trace (Eq. 1)
            </button>
            <button
              onClick={() => setActiveTab('rag')}
              className={`flex-1 py-2 px-2.5 rounded-lg font-bold transition-all text-center ${
                activeTab === 'rag'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              RAG ({analysisResult?.rag_references?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('synthesis')}
              className={`flex-1 py-2 px-2.5 rounded-lg font-bold transition-all text-center ${
                activeTab === 'synthesis'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Synthesis
            </button>
          </div>

          {/* Active Tab Content Area */}
          <div className="space-y-4 max-h-[850px] overflow-y-auto pr-1">
            {activeTab === 'findings' && (
              <FindingCards
                findings={analysisResult?.findings || []}
                selectedFindingUid={selectedFindingUid}
                onSelectFinding={setSelectedFindingUid}
                ragReferences={analysisResult?.rag_references || []}
              />
            )}

            {activeTab === 'trace' && (
              <OrchestrationTracePanel
                traces={analysisResult?.traces || []}
                detectedIntent={analysisResult?.detected_intent}
                totalLatencyMs={analysisResult?.total_latency_ms}
              />
            )}

            {activeTab === 'rag' && (
              <RAGPanel
                references={analysisResult?.rag_references || []}
                evidenceSupportScore={analysisResult?.evidence_support_score}
              />
            )}

            {activeTab === 'synthesis' && (
              <div className="command-panel p-5 space-y-4">
                <div className="flex items-center space-x-2 pb-3 border-b border-[#2B3A54]">
                  <Sparkles className="w-4 h-4 text-[#43D9F5]" />
                  <h3 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
                    5-Layer Epistemological Synthesis
                  </h3>
                </div>

                {analysisResult?.synthesis_text ? (
                  <div className="space-y-3 text-xs leading-relaxed font-sans text-[#B4C0D4]">
                    <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54]">
                      <span className="text-[11px] font-mono font-bold text-[#43D9F5] uppercase block mb-1">
                        Synthesized Executive Report:
                      </span>
                      <p className="whitespace-pre-line text-[#F1F5F9]">
                        {analysisResult.synthesis_text}
                      </p>
                    </div>

                    <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] text-[11px] font-mono space-y-1">
                      <div className="text-[#8998AF]">
                        Evidence Support Grounding: <strong className="text-[#34D399]">Verified against Knowledge Base</strong>
                      </div>
                      <div className="text-[#8998AF]">
                        Zero Fabrication Policy: <strong className="text-white">Strict Epistemological Boundaries Enforced</strong>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-[#8998AF] text-xs">
                    Run an inspection analysis to generate multi-agent synthesis.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
