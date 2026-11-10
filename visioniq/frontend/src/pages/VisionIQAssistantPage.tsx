import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { AssistantResponse, AssistantFAQItem } from '../types';
import {
  Bot,
  Send,
  Sparkles,
  BookOpen,
  Compass,
  AlertCircle,
  Loader2,
  HelpCircle,
  ShieldCheck,
  CheckCircle2,
  Copy,
  Check
} from 'lucide-react';

interface ChatEntry {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  data?: AssistantResponse;
}

export const VisionIQAssistantPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialQuery = searchParams.get('q');
  const navigate = useNavigate();

  const [faqs, setFaqs] = useState<AssistantFAQItem[]>([]);
  const [messages, setMessages] = useState<ChatEntry[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: `### Welcome to the VisionIQ Engineering Guide

I am your technical assistant for the **VisionIQ Adaptive Multi-Agent Framework** (Dr. Mahalingam College of Engineering and Technology / Anna University Final Year Project 23CSP701 / IEEE Access manuscript).

I operate in **Deterministic Engineering Mode** grounded strictly in verified repository code, model checkpoints, and regulatory standards. Select one of the canonical topics below or ask any question regarding pipeline operations, training, routing equations, or empirical evaluations.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [queryInput, setQueryInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchFaqs = async () => {
      try {
        const list = await api.getAssistantFaqs();
        setFaqs(list || []);
      } catch (err) {
        console.error("Failed to load FAQs", err);
      }
    };
    fetchFaqs();
  }, []);

  useEffect(() => {
    if (initialQuery) {
      handleSendQuery(initialQuery);
    }
  }, [initialQuery]);

  const handleSendQuery = async (text: string) => {
    const q = text.trim();
    if (!q || loading) return;

    const userEntry: ChatEntry = {
      id: `u-${Date.now()}`,
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userEntry]);
    setQueryInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await api.queryAssistant(q);
      const botEntry: ChatEntry = {
        id: `a-${Date.now()}`,
        sender: 'assistant',
        text: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        data: response
      };
      setMessages((prev) => [...prev, botEntry]);
    } catch (err: any) {
      console.error(err);
      setError("Failed to query VisionIQ Assistant. Please ensure backend service is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#43D9F5]/15 border border-[#43D9F5]/30 flex items-center justify-center text-[#43D9F5]">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              VisionIQ AI Guide & System Documentation
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Deterministic Engineering Knowledge • Codebase Specifications • Verifiable Equations & Checkpoints
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3.5 py-1.5 rounded-lg bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40 text-xs font-mono flex items-center space-x-2 font-bold">
            <ShieldCheck className="w-4 h-4 text-[#34D399]" />
            <span>Deterministic Grounding Active</span>
          </span>
        </div>
      </div>

      {/* Main 2-Column: Left Canonical Prompt Picker (4 Cols), Right Chat Thread (8 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Canonical Knowledge Topics */}
        <div className="lg:col-span-4 space-y-4">
          <div className="command-panel p-5 space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
              <HelpCircle className="w-4 h-4 text-[#43D9F5]" />
              <span>Canonical Engineering Questions</span>
            </h2>
            <p className="text-xs text-[#8998AF]">
              Click any question below to inspect exact equations, commands, and verified protocols:
            </p>

            <div className="space-y-2 pt-1">
              {[
                { id: "1", q: "Which analysis mode should I use?", category: "Operations" },
                { id: "2", q: "Why did VisionIQ select these agents?", category: "Orchestration" },
                { id: "3", q: "What does this confidence score mean?", category: "Explainability" },
                { id: "4", q: "How do I train the PPE model?", category: "Machine Learning" },
                { id: "5", q: "How can I add a new technical document to RAG?", category: "Knowledge Base" },
                { id: "6", q: "Why could the damage cause not be determined?", category: "Epistemic Uncertainty" },
                { id: "7", q: "How do I interpret the evaluation metrics?", category: "Scientific Evaluation" },
                { id: "8", q: "How do I export and understand the PDF report?", category: "Reporting" }
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleSendQuery(item.q)}
                  disabled={loading}
                  className="w-full text-left p-3 rounded-lg bg-[#10182B] hover:bg-[#1B2942] border border-[#2B3A54] hover:border-[#43D9F5] text-xs text-[#B4C0D4] hover:text-white transition-all space-y-1 group"
                >
                  <div className="text-[10px] font-mono text-[#43D9F5] font-bold uppercase">{item.category}</div>
                  <div className="font-sans font-medium text-xs group-hover:text-white text-[#F1F5F9]">{item.q}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Interactive Conversation Thread */}
        <div className="lg:col-span-8 command-panel flex flex-col h-[740px] shadow-2xl overflow-hidden">
          {/* Messages Scroll Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex flex-col ${m.sender === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[90%] rounded-2xl p-5 text-xs leading-relaxed space-y-3 ${
                    m.sender === 'user'
                      ? 'bg-[#1B2942] border border-[#43D9F5]/40 text-white shadow-md'
                      : 'bg-[#10182B] border border-[#2B3A54] text-[#F1F5F9] shadow-md'
                  }`}
                >
                  <div className="prose prose-invert prose-xs max-w-none font-sans whitespace-pre-wrap leading-relaxed">
                    {m.text}
                  </div>

                  {/* Navigation Suggestion Action */}
                  {m.data?.recommended_navigation && (
                    <div className="pt-3 border-t border-[#2B3A54] flex items-center justify-between">
                      <span className="text-[11px] text-[#8998AF] font-mono">Recommended Workspace:</span>
                      <button
                        onClick={() => navigate(m.data!.recommended_navigation!.path)}
                        className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#43D9F5]/20 hover:bg-[#43D9F5]/30 border border-[#43D9F5]/50 text-[#43D9F5] text-xs font-semibold transition-colors"
                      >
                        <Compass className="w-3.5 h-3.5" />
                        <span>{m.data.recommended_navigation.label}</span>
                      </button>
                    </div>
                  )}

                  {/* Sources Grounding */}
                  {m.data?.source_references && m.data.source_references.length > 0 && (
                    <div className="pt-2 text-[11px] font-mono text-[#8998AF] flex items-center space-x-2 flex-wrap">
                      <BookOpen className="w-3.5 h-3.5 text-[#43D9F5] flex-shrink-0" />
                      <span>Verified Sources:</span>
                      {m.data.source_references.map((src, sIdx) => (
                        <span key={sIdx} className="bg-[#151F35] px-2 py-0.5 rounded border border-[#2B3A54] text-[#B4C0D4]">
                          {src}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Follow-up Prompts */}
                  {m.data?.suggested_followups && m.data.suggested_followups.length > 0 && (
                    <div className="pt-2 border-t border-[#2B3A54]">
                      <span className="text-[10px] font-mono uppercase text-[#8998AF] block mb-1">Related Queries:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {m.data.suggested_followups.map((sug, sIdx) => (
                          <button
                            key={sIdx}
                            onClick={() => handleSendQuery(sug)}
                            className="px-2.5 py-1 rounded-md bg-[#151F35] hover:bg-[#1B2942] border border-[#2B3A54] hover:border-[#43D9F5] text-xs text-[#43D9F5] transition-colors"
                          >
                            {sug}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-2 mt-1.5 px-2 text-[11px] font-mono text-[#8998AF]">
                  <span>{m.timestamp}</span>
                  {m.sender === 'assistant' && (
                    <button
                      onClick={() => handleCopyText(m.text, m.id)}
                      className="hover:text-white"
                      title="Copy response"
                    >
                      {copiedId === m.id ? <Check className="w-3 h-3 text-[#34D399]" /> : <Copy className="w-3 h-3" />}
                    </button>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex items-center space-x-2.5 p-4 rounded-xl bg-[#10182B] border border-[#2B3A54] text-xs text-[#43D9F5] w-fit font-mono">
                <Loader2 className="w-4 h-4 animate-spin text-[#43D9F5]" />
                <span>Searching verified repository specifications & equations...</span>
              </div>
            )}

            {error && (
              <div className="p-3.5 rounded-lg bg-[#FB7185]/15 border border-[#FB7185]/40 text-xs text-[#FB7185] flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-[#FB7185]" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Input Bar */}
          <div className="p-4 border-t border-[#2B3A54] bg-[#0B1020]/95">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendQuery(queryInput);
              }}
              className="flex items-center space-x-3"
            >
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="Ask about model architectures, equations, training steps, or RAG..."
                className="flex-1 px-4 py-3 rounded-xl bg-[#10182B] border border-[#2B3A54] text-xs text-white placeholder-[#8998AF] focus:outline-none focus:border-[#43D9F5]"
              />
              <button
                type="submit"
                disabled={loading || !queryInput.trim()}
                className="px-6 py-3 rounded-xl bg-[#43D9F5] hover:bg-[#38bdf8] disabled:opacity-40 text-[#0B1020] font-bold text-xs shadow-md transition-colors flex items-center space-x-2"
              >
                <span>Send</span>
                <Send className="w-3.5 h-3.5 text-[#0B1020]" />
              </button>
            </form>
            <div className="mt-2 text-center text-[11px] text-[#8998AF] font-mono">
              Grounding: Dr. MCET / Anna University 23CSP701 • IEEE Access Manuscript Specification
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
