import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { HistorySession, AnalysisResponseData } from '../types';
import {
  History,
  Search,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Eye,
  RefreshCw,
  Filter,
  Camera,
  Video,
  Image as ImageIcon,
  Download,
  X,
  ExternalLink,
  ShieldCheck,
  Cpu,
  Layers,
  Copy,
  Check
} from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [sessions, setSessions] = useState<HistorySession[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [modalityFilter, setModalityFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');

  // Quick View Modal State
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionDetail, setSessionDetail] = useState<AnalysisResponseData | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [activeTab, setActiveTab] = useState<'findings' | 'traces' | 'rag' | 'visuals'>('findings');

  // PDF Regeneration Feedback
  const [regeneratingId, setRegeneratingId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getHistory({
        query: searchQuery.trim() || undefined,
        input_type: modalityFilter !== 'all' ? modalityFilter : undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        limit: 100,
        order: sortOrder
      });
      setSessions(data || []);
    } catch (e) {
      console.error('Failed to load history:', e);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, modalityFilter, statusFilter, sortOrder]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleOpenDetail = async (sessionId: string) => {
    setSelectedSessionId(sessionId);
    setLoadingDetail(true);
    setActiveTab('findings');
    try {
      const detail = await api.getSessionDetails(sessionId);
      setSessionDetail(detail);
    } catch (e) {
      console.error('Failed to load session details:', e);
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleCloseDetail = () => {
    setSelectedSessionId(null);
    setSessionDetail(null);
  };

  const handleRegeneratePdf = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setRegeneratingId(sessionId);
    try {
      await api.generatePdfReport(sessionId);
      await fetchHistory();
    } catch (err) {
      console.error('Failed to regenerate PDF:', err);
      alert('PDF regeneration failed. Please ensure the inspection session exists.');
    } finally {
      setRegeneratingId(null);
    }
  };

  const copyToClipboard = (text: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedId(text);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getModalityIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'video':
        return <Video className="w-3.5 h-3.5 text-[#FBBF24]" />;
      case 'camera':
        return <Camera className="w-3.5 h-3.5 text-[#34D399]" />;
      case 'image':
      default:
        return <ImageIcon className="w-3.5 h-3.5 text-[#43D9F5]" />;
    }
  };

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Page Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#43D9F5]/15 border border-[#43D9F5]/30 flex items-center justify-center text-[#43D9F5]">
            <History className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              Inspection History & Provenance Records
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              SQLite Audit Trail • Equation 1 Parameters • Real Execution Latencies • ReportLab PDF Archive
            </p>
          </div>
        </div>

        <button
          onClick={() => fetchHistory()}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] hover:bg-[#1B2942] text-xs font-mono text-[#B4C0D4] hover:text-white transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-[#43D9F5]' : ''}`} />
          <span>Refresh Records</span>
        </button>
      </div>

      {/* Control Filter Bar */}
      <div className="command-panel p-4 grid grid-cols-1 md:grid-cols-4 gap-3">
        {/* Search Input */}
        <div className="relative md:col-span-1">
          <Search className="w-4 h-4 text-[#8998AF] absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search query, intent, ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white placeholder-[#8998AF] focus:outline-none focus:border-[#43D9F5] font-mono"
          />
        </div>

        {/* Modality Filter */}
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-[#8998AF]">Mode:</span>
          <select
            value={modalityFilter}
            onChange={(e) => setModalityFilter(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white focus:outline-none focus:border-[#43D9F5] font-mono"
          >
            <option value="all">All Modalities</option>
            <option value="image">Image Inspection</option>
            <option value="video">Video Tracking</option>
            <option value="camera">Live Camera</option>
          </select>
        </div>

        {/* Status Filter */}
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-[#8998AF]">Status:</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white focus:outline-none focus:border-[#43D9F5] font-mono"
          >
            <option value="all">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        {/* Sort Order */}
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-[#8998AF]">Order:</span>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as 'desc' | 'asc')}
            className="w-full px-3 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white focus:outline-none focus:border-[#43D9F5] font-mono"
          >
            <option value="desc">Newest First (Desc)</option>
            <option value="asc">Oldest First (Asc)</option>
          </select>
        </div>
      </div>

      {/* Results Header Count */}
      <div className="flex items-center justify-between text-xs font-mono text-[#8998AF] px-1">
        <span>Showing {sessions.length} historical inspection session{sessions.length === 1 ? '' : 's'} in SQLite database</span>
      </div>

      {/* Main Table */}
      {loading ? (
        <div className="command-panel p-16 text-center text-xs font-mono text-[#8998AF]">
          <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-[#43D9F5]" />
          Querying SQLite inspection archives...
        </div>
      ) : sessions.length === 0 ? (
        <div className="command-panel p-16 text-center text-xs font-mono text-[#8998AF]">
          No matching inspection sessions found for current filters.
        </div>
      ) : (
        <div className="command-panel overflow-hidden">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-[#2B3A54] bg-[#10182B] text-[#8998AF]">
                <th className="py-3.5 px-4 font-semibold">Session UID</th>
                <th className="py-3.5 px-3 font-semibold">Modality</th>
                <th className="py-3.5 px-4 font-semibold">Date / Time (UTC)</th>
                <th className="py-3.5 px-4 font-semibold">Inspection Query</th>
                <th className="py-3.5 px-3 font-semibold">Detected Intent</th>
                <th className="py-3.5 px-3 text-center font-semibold">Latency</th>
                <th className="py-3.5 px-3 text-center font-semibold">Status</th>
                <th className="py-3.5 px-4 text-right font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#2B3A54]/70">
              {sessions.map((s) => (
                <tr
                  key={s.id}
                  onClick={() => handleOpenDetail(s.id)}
                  className="hover:bg-[#1B2942]/40 cursor-pointer transition-colors group"
                >
                  {/* Session UID */}
                  <td className="py-3 px-4 text-[#43D9F5] font-bold whitespace-nowrap">
                    <div className="flex items-center space-x-1.5">
                      <span>{s.id.slice(0, 10)}</span>
                      <button
                        onClick={(e) => copyToClipboard(s.id, e)}
                        title="Copy full UUID"
                        className="opacity-0 group-hover:opacity-100 text-[#8998AF] hover:text-white transition-opacity"
                      >
                        {copiedId === s.id ? (
                          <Check className="w-3.5 h-3.5 text-[#34D399]" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </td>

                  {/* Modality */}
                  <td className="py-3 px-3 whitespace-nowrap">
                    <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded bg-[#10182B] text-[#B4C0D4] border border-[#2B3A54] text-[11px]">
                      {getModalityIcon(s.input_type || 'image')}
                      <span className="capitalize">{s.input_type || 'image'}</span>
                    </span>
                  </td>

                  {/* Timestamp */}
                  <td className="py-3 px-4 text-[#8998AF] whitespace-nowrap text-[11px]">
                    {s.created_at ? new Date(s.created_at).toLocaleString() : 'N/A'}
                  </td>

                  {/* Query */}
                  <td className="py-3 px-4 text-white font-sans max-w-xs truncate font-medium" title={s.user_query}>
                    {s.user_query}
                  </td>

                  {/* Detected Intent */}
                  <td className="py-3 px-3 whitespace-nowrap">
                    <span className="px-2.5 py-1 rounded bg-[#10182B] text-[#B4C0D4] border border-[#2B3A54] text-[11px]">
                      {s.detected_intent || 'General'}
                    </span>
                  </td>

                  {/* Latency */}
                  <td className="py-3 px-3 text-center text-[#B4C0D4] whitespace-nowrap">
                    {s.total_latency_ms.toFixed(1)} ms
                  </td>

                  {/* Status */}
                  <td className="py-3 px-3 text-center whitespace-nowrap">
                    {s.status === 'completed' ? (
                      <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded text-[11px] bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/30 font-semibold">
                        <CheckCircle2 className="w-3 h-3" />
                        <span>Completed</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded text-[11px] bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/30 font-semibold">
                        <AlertTriangle className="w-3 h-3" />
                        <span>{s.status}</span>
                      </span>
                    )}
                  </td>

                  {/* Actions */}
                  <td className="py-3 px-4 text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={() => handleOpenDetail(s.id)}
                        title="Inspect session findings and traces"
                        className="p-1.5 rounded hover:bg-[#1B2942] text-[#8998AF] hover:text-[#43D9F5] transition-colors"
                      >
                        <Eye className="w-4 h-4" />
                      </button>

                      <a
                        href={api.getPdfReportUrl(s.id)}
                        download={`VisionIQ_Report_${s.id.slice(0, 8)}.pdf`}
                        target="_blank"
                        rel="noreferrer"
                        title="Download verified ReportLab PDF report"
                        className="inline-flex items-center space-x-1 px-3 py-1 rounded-lg border border-[#43D9F5]/50 bg-[#43D9F5]/15 hover:bg-[#43D9F5]/25 text-[#43D9F5] text-[11px] font-bold transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>PDF</span>
                      </a>

                      <button
                        onClick={(e) => handleRegeneratePdf(s.id, e)}
                        disabled={regeneratingId === s.id}
                        title="Re-synthesize PDF report with latest metrics"
                        className="p-1.5 rounded hover:bg-[#1B2942] text-[#8998AF] hover:text-white transition-colors"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${regeneratingId === s.id ? 'animate-spin text-[#FBBF24]' : ''}`} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Quick View Inspection Modal */}
      {selectedSessionId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-[#10182B] border border-[#2B3A54] rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-[#2B3A54] bg-[#151F35]">
              <div className="flex items-center space-x-3">
                <FileText className="w-5 h-5 text-[#43D9F5]" />
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="text-sm font-bold text-white font-mono">
                      Session #{selectedSessionId.slice(0, 12)}
                    </h3>
                    <span className="text-xs text-[#8998AF] font-mono">
                      ({sessionDetail?.input_type || 'image'})
                    </span>
                  </div>
                  <p className="text-xs text-[#B4C0D4] truncate max-w-md font-sans">
                    {sessionDetail?.user_query || 'Loading inspection parameters...'}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <a
                  href={api.getPdfReportUrl(selectedSessionId)}
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-[#43D9F5]/50 bg-[#43D9F5]/15 hover:bg-[#43D9F5]/25 text-[#43D9F5] text-xs font-mono font-bold transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download PDF</span>
                </a>
                <button
                  onClick={handleCloseDetail}
                  className="p-1.5 rounded-lg text-[#8998AF] hover:text-white hover:bg-[#1B2942] transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Navigation Tabs */}
            <div className="flex items-center px-6 border-b border-[#2B3A54] bg-[#0B1020] text-xs font-mono">
              <button
                onClick={() => setActiveTab('findings')}
                className={`py-3 px-4 border-b-2 font-semibold transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'findings'
                    ? 'border-[#43D9F5] text-[#43D9F5]'
                    : 'border-transparent text-[#8998AF] hover:text-white'
                }`}
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Findings ({sessionDetail?.findings?.length || 0})</span>
              </button>

              <button
                onClick={() => setActiveTab('traces')}
                className={`py-3 px-4 border-b-2 font-semibold transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'traces'
                    ? 'border-[#43D9F5] text-[#43D9F5]'
                    : 'border-transparent text-[#8998AF] hover:text-white'
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>Agent Traces ({sessionDetail?.traces?.length || 0})</span>
              </button>

              <button
                onClick={() => setActiveTab('rag')}
                className={`py-3 px-4 border-b-2 font-semibold transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'rag'
                    ? 'border-[#43D9F5] text-[#43D9F5]'
                    : 'border-transparent text-[#8998AF] hover:text-white'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>
                  RAG Citations ({((sessionDetail?.rag_references || sessionDetail?.rag_sources) || []).length})
                </span>
              </button>

              <button
                onClick={() => setActiveTab('visuals')}
                className={`py-3 px-4 border-b-2 font-semibold transition-colors flex items-center space-x-1.5 ${
                  activeTab === 'visuals'
                    ? 'border-[#43D9F5] text-[#43D9F5]'
                    : 'border-transparent text-[#8998AF] hover:text-white'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Visual Evidence</span>
              </button>
            </div>

            {/* Modal Content Body */}
            <div className="p-6 overflow-y-auto flex-1 space-y-4 font-mono text-xs">
              {loadingDetail ? (
                <div className="p-12 text-center text-[#8998AF]">
                  <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-[#43D9F5]" />
                  Loading session data from database...
                </div>
              ) : !sessionDetail ? (
                <div className="p-12 text-center text-[#8998AF]">
                  Failed to retrieve session details.
                </div>
              ) : (
                <>
                  {/* Tab 1: Findings */}
                  {activeTab === 'findings' && (
                    <div className="space-y-3">
                      {sessionDetail.findings && sessionDetail.findings.length > 0 ? (
                        sessionDetail.findings.map((f, idx) => (
                          <div
                            key={idx}
                            className="p-4 rounded-xl border border-[#2B3A54] bg-[#151F35] space-y-2"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <span className="px-2 py-0.5 rounded bg-[#10182B] text-[#43D9F5] font-bold border border-[#2B3A54]">
                                  {f.finding_uid}
                                </span>
                                <span className="text-white font-semibold text-sm">{f.label}</span>
                                <span className="text-[#8998AF] uppercase text-[10px]">
                                  ({f.agent_source})
                                </span>
                              </div>
                              <div className="flex items-center space-x-2">
                                {f.detection_confidence !== null && f.detection_confidence !== undefined && (
                                  <span className="text-[#34D399] font-bold">
                                    {(f.detection_confidence * 100).toFixed(1)}% conf
                                  </span>
                                )}
                                {f.human_review_required && (
                                  <span className="px-2 py-0.5 rounded bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/30 text-[10px] font-bold">
                                    HUMAN REVIEW
                                  </span>
                                )}
                              </div>
                            </div>

                            <p className="text-[#B4C0D4] font-sans text-xs">
                              <strong className="text-white">Evidence:</strong> {f.observed_evidence}
                            </p>

                            {f.root_cause_hypothesis && (
                              <p className="text-[#8998AF] font-sans text-xs">
                                <strong className="text-[#FBBF24]">Root Cause Hypothesis:</strong> {f.root_cause_hypothesis}
                              </p>
                            )}

                            {f.recommendation && (
                              <p className="text-[#8998AF] font-sans text-xs">
                                <strong className="text-[#34D399]">Recommendation:</strong> {f.recommendation}
                              </p>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="p-8 text-center text-[#8998AF]">
                          No discrete findings recorded for this query.
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tab 2: Traces */}
                  {activeTab === 'traces' && (
                    <div className="space-y-3">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="border-b border-[#2B3A54] text-[#8998AF] bg-[#10182B]">
                              <th className="py-2.5 px-3">Agent</th>
                              <th className="py-2.5 px-2">R(a,q)</th>
                              <th className="py-2.5 px-2">E(a)</th>
                              <th className="py-2.5 px-2">C(a)</th>
                              <th className="py-2.5 px-2">L(a)</th>
                              <th className="py-2.5 px-3">Score S(a,q)</th>
                              <th className="py-2.5 px-3">Decision</th>
                              <th className="py-2.5 px-3 text-right">Runtime</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-[#2B3A54]">
                            {sessionDetail.traces && sessionDetail.traces.map((t, i) => (
                              <tr key={i} className="hover:bg-[#1B2942]/30">
                                <td className="py-2 px-3 text-white font-bold">{t.agent_name}</td>
                                <td className="py-2 px-2 text-[#B4C0D4]">{t.relevance_score.toFixed(2)}</td>
                                <td className="py-2 px-2 text-[#B4C0D4]">{t.evidence_capability.toFixed(2)}</td>
                                <td className="py-2 px-2 text-[#B4C0D4]">{t.normalized_cost.toFixed(2)}</td>
                                <td className="py-2 px-2 text-[#B4C0D4]">{t.expected_latency.toFixed(2)}</td>
                                <td className="py-2 px-3 text-[#43D9F5] font-bold">
                                  {t.final_routing_score.toFixed(3)}
                                </td>
                                <td className="py-2 px-3">
                                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                    t.decision === 'SELECTED'
                                      ? 'bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/30'
                                      : 'bg-[#10182B] text-[#8998AF] border border-[#2B3A54]'
                                  }`}>
                                    {t.decision}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-right text-[#B4C0D4]">
                                  {t.actual_latency_ms.toFixed(1)} ms
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Tab 3: RAG Citations */}
                  {activeTab === 'rag' && (
                    <div className="space-y-3">
                      {((sessionDetail.rag_references || sessionDetail.rag_sources) || []).length > 0 ? (
                        ((sessionDetail.rag_references || sessionDetail.rag_sources) || []).map((r, i) => (
                          <div
                            key={i}
                            className="p-4 rounded-xl border border-[#2B3A54] bg-[#151F35] space-y-1.5"
                          >
                            <div className="flex items-center justify-between text-[#43D9F5] font-bold">
                              <span>[{i + 1}] {r.document_title}</span>
                              <span className="text-[#34D399] font-normal">
                                Match: {(r.similarity_score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <p className="text-xs text-[#B4C0D4] font-sans italic leading-relaxed">
                              "{r.excerpt}"
                            </p>
                            <div className="text-[11px] text-[#8998AF]">
                              Section: <span className="text-white">{r.section_reference}</span> | Source: <span className="text-[#43D9F5]">{r.source_uri || 'Internal Archive'}</span>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="p-8 text-center text-[#8998AF]">
                          No regulatory RAG citations retrieved for this inspection session.
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tab 4: Visuals */}
                  {activeTab === 'visuals' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="text-xs text-[#8998AF] font-bold">Raw Input Visual:</div>
                        {sessionDetail.input_url ? (
                          <img
                            src={sessionDetail.input_url}
                            alt="Raw Sensor Input"
                            className="rounded-lg border border-[#2B3A54] max-h-72 w-full object-contain bg-black/40"
                          />
                        ) : (
                          <div className="p-12 text-center border border-[#2B3A54] rounded-lg text-[#8998AF]">
                            No input image preview URL available.
                          </div>
                        )}
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs text-[#8998AF] font-bold">Annotated Evidence Overlays:</div>
                        {sessionDetail.annotated_url ? (
                          <img
                            src={sessionDetail.annotated_url}
                            alt="Annotated Detections"
                            className="rounded-lg border border-[#2B3A54] max-h-72 w-full object-contain bg-black/40"
                          />
                        ) : (
                          <div className="p-12 text-center border border-[#2B3A54] rounded-lg text-[#8998AF]">
                            Annotated overlay view not generated for this session.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
