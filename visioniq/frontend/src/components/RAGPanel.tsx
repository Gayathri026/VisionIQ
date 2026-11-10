import React, { useState, useEffect } from 'react';
import { RAGReference } from '../types';
import { api } from '../services/api';
import { BookOpen, ExternalLink, Bookmark, Search, Database, ChevronDown, ChevronUp, Shield, FileText, CheckCircle2 } from 'lucide-react';

interface Props {
  references: RAGReference[];
  evidenceSupportScore?: number;
}

export const RAGPanel: React.FC<Props> = ({ references, evidenceSupportScore }) => {
  const [showDocExplorer, setShowDocExplorer] = useState(false);
  const [indexedDocs, setIndexedDocs] = useState<any[]>([]);
  const [adhocQuery, setAdhocQuery] = useState('');
  const [adhocResults, setAdhocResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [expandedChunkId, setExpandedChunkId] = useState<number | null>(null);

  useEffect(() => {
    if (showDocExplorer && indexedDocs.length === 0) {
      api.getRagDocuments()
        .then((docs) => setIndexedDocs(docs || []))
        .catch((err) => console.error("Failed to load indexed documents:", err));
    }
  }, [showDocExplorer]);

  const handleSearchKnowledgeBase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!adhocQuery.trim()) return;
    setSearching(true);
    try {
      const data = await api.queryRag(adhocQuery, 4, 0.25);
      setAdhocResults(data.passages || []);
    } catch (err) {
      console.error("Ad-hoc RAG query failed:", err);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="command-panel rounded-2xl p-6 sm:p-7 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#2B3A54]">
        <div className="flex items-center space-x-2.5">
          <Bookmark className="w-5 h-5 text-[#43D9F5]" />
          <h3 className="text-lg font-bold text-[#F1F5F9]">
            RAG Knowledge Grounding & Standards Citations
          </h3>
        </div>

        <div className="flex items-center space-x-3">
          {evidenceSupportScore !== undefined && (
            <span className="text-xs font-mono font-bold text-[#34D399] bg-[#34D399]/15 border border-[#34D399]/40 px-3 py-1 rounded-lg">
              Groundedness: {(evidenceSupportScore * 100).toFixed(0)}%
            </span>
          )}
          <button
            onClick={() => setShowDocExplorer(!showDocExplorer)}
            className="flex items-center space-x-1.5 text-xs font-mono bg-[#0B1020] hover:bg-[#1B2942] text-[#B4C0D4] hover:text-[#F1F5F9] px-3.5 py-1.5 rounded-lg border border-[#2B3A54] transition-all cursor-pointer"
          >
            <Database className="w-3.5 h-3.5 text-[#43D9F5]" />
            <span>{showDocExplorer ? "Hide Vector Catalog" : "Inspect Vector Catalog"}</span>
            {showDocExplorer ? <ChevronUp className="w-3.5 h-3.5 ml-0.5" /> : <ChevronDown className="w-3.5 h-3.5 ml-0.5" />}
          </button>
        </div>
      </div>

      {/* Epistemological Layer Disclaimer */}
      <div className="p-4 rounded-xl bg-[#818CF8]/10 border border-[#818CF8]/30 text-xs sm:text-sm text-[#B4C0D4] flex items-start space-x-3">
        <Shield className="w-5 h-5 text-[#818CF8] flex-shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <strong className="text-[#F1F5F9] font-semibold">Epistemological Layer Separation:</strong> Retrieved standards represent authoritative technical specifications (OSHA 1910, ASTM A568/E165, ISO 13849). While they provide official regulatory criteria, they provide theoretical guidance and do not constitute physical proof without non-destructive testing (NDT).
        </div>
      </div>

      {/* Knowledge Store Explorer Dropdown */}
      {showDocExplorer && (
        <div className="p-5 rounded-xl bg-[#0B1020] border border-[#2B3A54] space-y-5 animate-in fade-in duration-200">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-mono font-bold text-[#F1F5F9] uppercase tracking-wider flex items-center space-x-2">
              <FileText className="w-4 h-4 text-[#FBBF24]" />
              <span>Indexed Standards & Specifications ({indexedDocs.length})</span>
            </h4>
            <span className="text-xs font-mono text-[#34D399]">Hybrid TF-IDF + BM25 Vector Store</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {indexedDocs.map((doc: any, idx: number) => (
              <div key={idx} className="p-3.5 rounded-xl bg-[#151F35] border border-[#2B3A54] text-xs font-mono">
                <div className="font-bold text-[#F1F5F9] truncate" title={doc.title}>{doc.title}</div>
                <div className="text-xs text-[#B4C0D4] mt-1.5 flex justify-between">
                  <span>{doc.format?.toUpperCase()} format</span>
                  <span>{doc.total_chunks || 1} chunks indexed</span>
                </div>
              </div>
            ))}
          </div>

          {/* Ad-Hoc Knowledge Search Form */}
          <form onSubmit={handleSearchKnowledgeBase} className="flex items-center space-x-2.5 pt-3 border-t border-[#2B3A54]">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-[#64748B] absolute left-3.5 top-3" />
              <input
                type="text"
                placeholder="Search standards directly (e.g., 'ANSI Z89.1 hard hats', 'ASTM A568 scratches', 'ISO 13849')..."
                value={adhocQuery}
                onChange={(e) => setAdhocQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-xl bg-[#151F35] border border-[#2B3A54] text-sm text-[#F1F5F9] placeholder-[#64748B] focus:outline-none focus:border-[#43D9F5]"
              />
            </div>
            <button
              type="submit"
              disabled={searching || !adhocQuery.trim()}
              className="px-4 py-2 rounded-xl bg-[#43D9F5] text-[#0B1020] text-sm font-bold font-mono hover:opacity-90 disabled:opacity-50 transition-all cursor-pointer"
            >
              {searching ? "Searching..." : "Search"}
            </button>
          </form>

          {/* Ad-hoc search results */}
          {adhocResults.length > 0 && (
            <div className="space-y-2.5 pt-2">
              <span className="text-xs font-mono text-[#43D9F5] font-semibold block">Top Retrieved Passages:</span>
              {adhocResults.map((p, pIdx) => (
                <div key={pIdx} className="p-3.5 rounded-xl bg-[#151F35] border border-[#43D9F5]/30 text-xs sm:text-sm">
                  <div className="flex justify-between font-mono text-xs mb-1.5">
                    <span className="text-[#F1F5F9] font-bold">{p.document_title} — {p.section_reference}</span>
                    <span className="text-[#34D399] font-bold">{(p.similarity_score * 100).toFixed(1)}% Match</span>
                  </div>
                  <p className="text-xs sm:text-sm text-[#B4C0D4] font-sans italic">"{p.excerpt}"</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Main Grounding References */}
      {references.length === 0 ? (
        <div className="text-center py-8 text-[#B4C0D4] bg-[#0B1020]/40 rounded-xl border border-[#2B3A54]/60">
          <BookOpen className="w-8 h-8 mx-auto mb-2.5 text-[#64748B]" />
          <p className="text-sm font-medium">No external regulatory RAG citations required for this query.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {references.map((ref, idx) => (
            <div key={idx} className="p-4 sm:p-5 rounded-xl bg-[#0B1020]/80 border border-[#2B3A54] space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="text-base font-bold text-[#F1F5F9]">{ref.document_title}</h4>
                  <span className="text-xs font-mono text-[#43D9F5] font-semibold">{ref.section_reference}</span>
                </div>
                <div className="flex items-center space-x-2.5">
                  <span className="text-xs font-mono font-bold text-[#34D399] bg-[#34D399]/15 border border-[#34D399]/40 px-2.5 py-0.5 rounded-lg">
                    {(ref.similarity_score * 100).toFixed(1)}% Match
                  </span>
                  <button
                    onClick={() => setExpandedChunkId(expandedChunkId === idx ? null : idx)}
                    className="text-[#B4C0D4] hover:text-[#F1F5F9] p-1.5 rounded-lg hover:bg-[#151F35] transition-colors"
                    title="Toggle passage inspector"
                  >
                    {expandedChunkId === idx ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <p className={`text-sm text-[#F1F5F9] italic leading-relaxed bg-[#151F35] p-3.5 rounded-xl border border-[#2B3A54] ${
                expandedChunkId === idx ? '' : 'line-clamp-3'
              }`}>
                "{ref.excerpt}"
              </p>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs font-mono text-[#B4C0D4] gap-2 pt-1">
                {ref.source_uri ? (
                  <a
                    href={ref.source_uri}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center space-x-1.5 text-[#43D9F5] hover:underline"
                  >
                    <span>Official Standards Reference</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                ) : (
                  <span className="text-[#64748B]">Internal Standard Document</span>
                )}

                <div>
                  Epistemic Status: <span className="text-[#F1F5F9] font-semibold">Normative Specification</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
