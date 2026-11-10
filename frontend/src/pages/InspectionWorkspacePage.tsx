import React, { useState } from 'react';
import { api } from '../services/api';
import { AnalysisResponseData } from '../types';
import { EvidenceWorkspace } from '../components/EvidenceWorkspace';
import { OrchestrationTracePanel } from '../components/OrchestrationTracePanel';
import { FindingCards } from '../components/FindingCards';
import { RAGPanel } from '../components/RAGPanel';
import { Upload, Sparkles, AlertCircle, Loader2 } from 'lucide-react';

const SAMPLE_QUERIES = [
  "Check if workers are wearing safety helmets and reflective vests",
  "Is there any crack, scratch, or surface defect on the metal plate?",
  "What is the machine serial number written on the label?",
  "Why did the hydraulic press halt suddenly during production?"
];

export const InspectionWorkspacePage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponseData | null>(null);
  const [selectedFindingUid, setSelectedFindingUid] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleRunAnalysis = async () => {
    if (!selectedFile) {
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
      const data = await api.analyzeImage(selectedFile, query);
      setAnalysisResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Analysis request failed. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-6">
      {/* Query Submission Box */}
      <div className="tech-panel rounded-xl p-5 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* File Picker */}
          <div className="md:col-span-4">
            <label className="block text-xs font-semibold text-graphite-300 uppercase tracking-wider font-mono mb-1.5">
              Input Inspection Image
            </label>
            <div className="relative">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="flex items-center justify-between w-full px-3.5 py-2.5 rounded-lg border border-graphite-700 bg-graphite-800/80 hover:bg-graphite-800 text-xs font-mono text-graphite-300 cursor-pointer transition-colors"
              >
                <div className="flex items-center space-x-2 truncate">
                  <Upload className="w-4 h-4 text-accent-cyan flex-shrink-0" />
                  <span className="truncate">{selectedFile ? selectedFile.name : "Select factory image..."}</span>
                </div>
                <span className="text-[10px] bg-graphite-700 px-2 py-0.5 rounded text-white font-sans">Browse</span>
              </label>
            </div>
          </div>

          {/* Query Input */}
          <div className="md:col-span-6">
            <label className="block text-xs font-semibold text-graphite-300 uppercase tracking-wider font-mono mb-1.5">
              Natural Language Inspection Question
            </label>
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about PPE compliance, defects, OCR text, or root causes..."
                className="w-full px-3.5 py-2.5 rounded-lg border border-graphite-700 bg-graphite-800/80 focus:border-accent-cyan focus:outline-none text-xs font-sans text-white placeholder-graphite-500"
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="md:col-span-2 flex items-end">
            <button
              onClick={handleRunAnalysis}
              disabled={loading}
              className="w-full h-[42px] mt-auto flex items-center justify-center space-x-2 bg-gradient-to-r from-accent-blue to-accent-cyan hover:opacity-90 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-all shadow-md shadow-accent-blue/20"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Orchestrating...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute VisionIQ</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Sample Queries Quick Selection */}
        <div className="mt-3 pt-3 border-t border-graphite-800 flex items-center space-x-2 overflow-x-auto text-[11px] font-mono text-graphite-400">
          <span className="whitespace-nowrap">Sample Prompts:</span>
          {SAMPLE_QUERIES.map((sq, idx) => (
            <button
              key={idx}
              onClick={() => setQuery(sq)}
              className="px-2 py-0.5 rounded bg-graphite-800/80 hover:bg-graphite-700 text-graphite-300 hover:text-white transition-colors truncate max-w-xs border border-graphite-700/60"
            >
              {sq}
            </button>
          ))}
        </div>

        {/* Error Notification */}
        {error && (
          <div className="mt-3 p-3 rounded-lg bg-red-950/40 border border-red-500/40 text-xs text-red-300 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Visual Canvas & Dynamic Orchestration Trace */}
        <div className="lg:col-span-8 space-y-6">
          <EvidenceWorkspace
            originalUrl={previewUrl || analysisResult?.input_url}
            annotatedUrl={analysisResult?.annotated_url}
            pdfUrl={analysisResult?.pdf_url}
            findings={analysisResult?.findings || []}
            selectedFindingUid={selectedFindingUid}
            onSelectFinding={setSelectedFindingUid}
          />

          {analysisResult && (
            <OrchestrationTracePanel
              traces={analysisResult.traces}
              detectedIntent={analysisResult.detected_intent}
              totalLatencyMs={analysisResult.total_latency_ms}
            />
          )}
        </div>

        {/* Right Column: Finding Cards & RAG Standards */}
        <div className="lg:col-span-4 space-y-6">
          {analysisResult && (
            <div className="tech-panel rounded-xl p-4 bg-gradient-to-br from-graphite-800/80 to-graphite-900 border-accent-blue/30">
              <h4 className="text-xs font-semibold text-accent-cyan uppercase tracking-wider font-mono mb-1">
                Executive Synthesis
              </h4>
              <p className="text-xs text-graphite-200 leading-relaxed font-sans">
                {analysisResult.synthesis_text}
              </p>
            </div>
          )}

          <FindingCards
            findings={analysisResult?.findings || []}
            selectedFindingUid={selectedFindingUid}
            onSelectFinding={setSelectedFindingUid}
          />

          {analysisResult && analysisResult.rag_references && (
            <RAGPanel
              references={analysisResult.rag_references}
              evidenceSupportScore={analysisResult.evidence_support_score}
            />
          )}
        </div>
      </div>
    </div>
  );
};
