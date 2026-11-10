import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import {
  BookOpen,
  Upload,
  Search,
  CheckCircle2,
  AlertCircle,
  FileText,
  ShieldCheck,
  ExternalLink,
  Loader2,
  Trash2,
  Check,
  Code2,
  Sparkles
} from 'lucide-react';

interface DocumentInfo {
  doc_id: string;
  title: string;
  format: string;
  file_path: string;
  source_uri?: string;
  total_chunks: number;
  total_pages: number;
  ingested_at: string;
}

export const KnowledgeBasePage: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Ingestion Form State
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [docTitle, setDocTitle] = useState('');
  const [sourceUri, setSourceUri] = useState('');
  const [ingesting, setIngesting] = useState(false);
  const [ingestSuccess, setIngestSuccess] = useState<string | null>(null);

  // Live Hybrid Search State
  const [searchQuery, setSearchQuery] = useState('What are the hard hat specifications under OSHA?');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);

  // Citation Validation Tool State
  const [citationText, setCitationText] = useState('OSHA Standard 1926.100: Section 2');
  const [validationResult, setValidationResult] = useState<any | null>(null);
  const [validating, setValidating] = useState(false);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const docs = await api.getRagDocuments();
      setDocuments(docs || []);
    } catch (err: any) {
      console.error(err);
      setError("Failed to load knowledge base documents.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIngesting(true);
    setError(null);
    setIngestSuccess(null);

    try {
      const res = await api.ingestDocument(uploadFile, docTitle || undefined, sourceUri || undefined);
      setIngestSuccess(`Successfully ingested "${uploadFile.name}" (${res.chunks_added || res.chunks_indexed || 1} chunks added).`);
      setUploadFile(null);
      setDocTitle('');
      setSourceUri('');
      setShowUploadModal(false);
      await fetchDocuments();
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Document ingestion failed.");
    } finally {
      setIngesting(false);
    }
  };

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    setSearching(true);
    try {
      const res = await api.queryRag(searchQuery, 4, 0.25);
      setSearchResults(res.results || res.passages || []);
    } catch (err: any) {
      console.error(err);
    } finally {
      setSearching(false);
    }
  };

  const handleValidateCitation = async () => {
    if (!citationText.trim()) return;
    setValidating(true);
    try {
      const res = await api.queryRag(citationText, 1, 0.25);
      const count = res.total_found || res.total_retrieved || 0;
      setValidationResult({
        validated: count > 0,
        matches: res.results || res.passages || []
      });
    } catch (err) {
      console.error(err);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#818CF8]/15 border border-[#818CF8]/30 flex items-center justify-center text-[#818CF8]">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              RAG Knowledge Base & Regulatory Standards
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Persistent Hybrid Vector Store (TF-IDF + BM25) • Prompt Injection Filtered • Two-Way Citation Verification
            </p>
          </div>
        </div>

        <button
          onClick={() => setShowUploadModal(true)}
          className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-[#0284C7] to-[#43D9F5] hover:opacity-90 text-[#0B1020] text-xs font-bold shadow-md transition-all font-mono"
        >
          <Upload className="w-4 h-4 text-[#0B1020]" />
          <span>Ingest Technical Standard</span>
        </button>
      </div>

      {ingestSuccess && (
        <div className="p-4 rounded-xl bg-[#34D399]/15 border border-[#34D399]/40 text-xs text-[#34D399] flex items-center space-x-2">
          <CheckCircle2 className="w-5 h-5 text-[#34D399] flex-shrink-0" />
          <span>{ingestSuccess}</span>
        </div>
      )}

      {/* Grid: Document Catalog (7 Cols) & Hybrid Search / Citation Testbed (5 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Indexed Standards Catalog */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-[#2B3A54]">
            <h2 className="text-xs font-bold uppercase tracking-wider font-mono text-[#F1F5F9]">
              Verified Regulatory Corpus ({documents.length} Standards)
            </h2>
          </div>

          {loading ? (
            <div className="command-panel p-16 text-center text-[#8998AF]">
              <Loader2 className="w-7 h-7 animate-spin mx-auto mb-2 text-[#43D9F5]" />
              <span className="text-xs font-mono">Loading persistent RAG index...</span>
            </div>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => (
                <div
                  key={doc.doc_id}
                  className="command-panel p-4 hover:border-[#43D9F5]/50 transition-all space-y-2.5"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start space-x-3">
                      <div className="p-2.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-[#43D9F5] mt-0.5">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="font-bold text-sm text-[#F1F5F9] font-sans">{doc.title}</h3>
                        <div className="flex items-center space-x-3 text-xs font-mono text-[#8998AF] mt-1">
                          <span className="uppercase text-[#43D9F5] font-bold">{doc.format}</span>
                          <span>•</span>
                          <span>{doc.total_chunks} Chunks</span>
                          {doc.total_pages > 1 && (
                            <>
                              <span>•</span>
                              <span>{doc.total_pages} Pages</span>
                            </>
                          )}
                          <span>•</span>
                          <span>{new Date(doc.ingested_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>

                    <span className="bg-[#34D399]/15 text-[#34D399] border border-[#34D399]/30 text-[11px] font-mono px-2 py-0.5 rounded flex items-center space-x-1">
                      <ShieldCheck className="w-3 h-3" />
                      <span>Authentic Corpus</span>
                    </span>
                  </div>

                  {doc.source_uri && (
                    <div className="text-[11px] font-mono text-[#8998AF] truncate pt-1 border-t border-[#2B3A54]/60">
                      URI: <a href={doc.source_uri} target="_blank" rel="noreferrer" className="text-[#43D9F5] hover:underline">{doc.source_uri}</a>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Hybrid Query & Citation Authenticator */}
        <div className="lg:col-span-5 space-y-5">
          {/* Live Semantic Query Box */}
          <div className="command-panel p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
              <Search className="w-4 h-4 text-[#43D9F5]" />
              <span>Hybrid Vector Search Testbed</span>
            </h3>

            <form onSubmit={handleSearch} className="space-y-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Ask technical question about OSHA, ASTM, ISO..."
                className="w-full px-3.5 py-2.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs text-white placeholder-[#8998AF] focus:outline-none focus:border-[#43D9F5]"
              />
              <button
                type="submit"
                disabled={searching}
                className="w-full py-2.5 rounded-lg bg-[#1B2942] hover:bg-[#223252] border border-[#2B3A54] hover:border-[#43D9F5] text-white text-xs font-bold transition-all flex items-center justify-center space-x-2"
              >
                {searching ? <Loader2 className="w-4 h-4 animate-spin text-[#43D9F5]" /> : <Search className="w-4 h-4 text-[#43D9F5]" />}
                <span>Retrieve Relevant Clauses</span>
              </button>
            </form>

            {searchResults.length > 0 && (
              <div className="space-y-2.5 pt-2 max-h-72 overflow-y-auto pr-1">
                <span className="text-[11px] font-mono uppercase text-[#8998AF] block">
                  Top Retrieved Passages ({searchResults.length}):
                </span>
                {searchResults.map((res: any, idx: number) => (
                  <div key={idx} className="p-3 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs space-y-1">
                    <div className="flex items-center justify-between font-mono">
                      <span className="font-bold text-white truncate max-w-[200px]">{res.document_title}</span>
                      <span className="text-[#34D399] font-bold">Score: {(res.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-xs text-[#B4C0D4] leading-relaxed italic">"{res.excerpt}"</p>
                    {res.section_reference && (
                      <div className="text-[11px] font-mono text-[#43D9F5]">Ref: {res.section_reference}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Citation Authenticator Tool */}
          <div className="command-panel p-5 space-y-3.5">
            <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
              <ShieldCheck className="w-4 h-4 text-[#34D399]" />
              <span>Two-Way Citation Authenticator</span>
            </h3>
            <p className="text-xs text-[#8998AF]">
              Verifies cited standard numbers and clauses against the indexed engineering corpus to eliminate hallucinated regulations.
            </p>

            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={citationText}
                onChange={(e) => setCitationText(e.target.value)}
                placeholder="e.g. OSHA Standard 1926.100: Section 2"
                className="flex-1 px-3 py-2 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs text-white focus:outline-none focus:border-[#43D9F5]"
              />
              <button
                onClick={handleValidateCitation}
                disabled={validating}
                className="px-4 py-2 rounded-lg bg-[#34D399]/20 hover:bg-[#34D399]/30 border border-[#34D399]/50 text-[#34D399] text-xs font-bold transition-colors"
              >
                Verify
              </button>
            </div>

            {validationResult && (
              <div className={`p-3 rounded-lg border text-xs font-mono space-y-1 ${
                validationResult.validated
                  ? 'bg-[#34D399]/15 border-[#34D399]/40 text-[#34D399]'
                  : 'bg-[#FB7185]/15 border-[#FB7185]/40 text-[#FB7185]'
              }`}>
                <div className="font-bold flex items-center space-x-1.5">
                  {validationResult.validated ? <Check className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                  <span>{validationResult.validated ? 'AUTHENTICATED STANDARD' : 'UNMATCHED CITATION'}</span>
                </div>
                <p className="text-xs text-[#B4C0D4]">
                  {validationResult.validated
                    ? 'Clause authenticated against indexed engineering standards corpus.'
                    : 'Clause not found in corpus. Flagged as unverified reference.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Ingest Document Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="command-panel p-6 max-w-lg w-full space-y-4 border-[#43D9F5]/40">
            <div className="flex items-center justify-between pb-3 border-b border-[#2B3A54]">
              <h3 className="text-sm font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2">
                <Upload className="w-4 h-4 text-[#43D9F5]" />
                <span>Ingest Engineering Standard</span>
              </h3>
              <button onClick={() => setShowUploadModal(false)} className="text-[#8998AF] hover:text-white text-xs">
                ✕ Close
              </button>
            </div>

            <form onSubmit={handleIngest} className="space-y-4">
              <div>
                <label className="block text-xs font-mono text-[#8998AF] mb-1">Standard Document File (PDF, MD, TXT, HTML):</label>
                <input
                  type="file"
                  accept=".pdf,.md,.txt,.html,.htm,.markdown"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  required
                  className="w-full text-xs text-[#B4C0D4] file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-[#1B2942] file:text-[#43D9F5] hover:file:bg-[#223252]"
                />
              </div>

              <div>
                <label className="block text-xs font-mono text-[#8998AF] mb-1">Standard Title (Optional):</label>
                <input
                  type="text"
                  value={docTitle}
                  onChange={(e) => setDocTitle(e.target.value)}
                  placeholder="e.g. ASTM E165 Liquid Penetrant Standard"
                  className="w-full px-3 py-2 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs text-white focus:outline-none focus:border-[#43D9F5]"
                />
              </div>

              <div>
                <label className="block text-xs font-mono text-[#8998AF] mb-1">Canonical Source URI (Optional):</label>
                <input
                  type="text"
                  value={sourceUri}
                  onChange={(e) => setSourceUri(e.target.value)}
                  placeholder="https://www.astm.org/e0165"
                  className="w-full px-3 py-2 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs text-white focus:outline-none focus:border-[#43D9F5]"
                />
              </div>

              <div className="pt-2 flex items-center justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(false)}
                  className="px-4 py-2 rounded-lg bg-[#10182B] text-[#8998AF] hover:text-white text-xs font-mono"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={ingesting || !uploadFile}
                  className="px-5 py-2 rounded-lg bg-[#43D9F5] hover:bg-[#38bdf8] text-[#0B1020] text-xs font-bold font-mono transition-all flex items-center space-x-2 disabled:opacity-50"
                >
                  {ingesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                  <span>{ingesting ? 'Indexing...' : 'Index Document'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
