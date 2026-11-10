import React, { useState } from 'react';
import { api } from '../services/api';
import { AssistantResponse, AgentTrace } from '../types';
import { X, Send, Bot, Sparkles, Compass, BookOpen, AlertCircle, Loader2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AssistantDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  activeSessionId?: string | null;
  activeTrace?: AgentTrace[] | null;
  activeFindingUid?: string | null;
}

interface ChatMessage {
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  data?: AssistantResponse;
}

const CANONICAL_QUESTIONS = [
  "Which analysis mode should I use?",
  "Why did VisionIQ select these agents?",
  "What does this confidence score mean?",
  "How do I train the PPE model?",
  "How can I add a new technical document to RAG?",
  "Why could the damage cause not be determined?",
  "How do I interpret the evaluation metrics?",
  "How do I export and understand the PDF report?"
];

export const AssistantDrawer: React.FC<AssistantDrawerProps> = ({
  isOpen,
  onClose,
  activeSessionId,
  activeTrace,
  activeFindingUid
}) => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      sender: 'assistant',
      text: "Hello! I am your **VisionIQ Technical Guide**.\n\nI provide operational guidance, explain multi-agent orchestration traces, and cite relevant regulatory standards (OSHA, ASTM, ISO) without speculative hallucination.\n\nSelect a common technical question below or ask me about your current inspection session.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSend = async (queryToSend: string) => {
    const q = queryToSend.trim();
    if (!q || loading) return;

    const userMsg: ChatMessage = {
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery('');
    setLoading(true);
    setError(null);

    try {
      const response = await api.queryAssistant(
        q,
        activeSessionId || undefined,
        activeTrace || undefined,
        activeFindingUid || undefined
      );

      const botMsg: ChatMessage = {
        sender: 'assistant',
        text: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        data: response
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err: any) {
      console.error(err);
      setError("Failed to query VisionIQ Assistant. Please verify that the FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleNavigate = (path: string) => {
    navigate(path);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/70 backdrop-blur-sm flex justify-end">
      <div className="w-full max-w-xl bg-[#0B1020] border-l border-[#2B3A54] h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-300">
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-[#2B3A54] bg-[#10182B] flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-[#43D9F5]/20 border border-[#43D9F5]/40 flex items-center justify-center text-[#43D9F5]">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="font-bold text-base text-[#F1F5F9]">VisionIQ AI Guide</h3>
                <span className="text-xs font-mono bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/30 px-2 py-0.5 rounded-full">
                  Deterministic
                </span>
              </div>
              <p className="text-xs text-[#B4C0D4] font-mono mt-0.5">
                {activeSessionId ? `Session ${activeSessionId.slice(0, 8)}... Grounding` : 'General Engineering Consultation'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-[#B4C0D4] hover:text-[#F1F5F9] hover:bg-[#151F35] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Canonical Question Chips */}
        <div className="p-3.5 border-b border-[#2B3A54] bg-[#0E1526]">
          <p className="text-xs font-mono uppercase text-[#B4C0D4] tracking-wider mb-2 font-semibold">
            Suggested Engineering Inquiries:
          </p>
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
            {CANONICAL_QUESTIONS.map((cq, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(cq)}
                disabled={loading}
                className="whitespace-nowrap px-3 py-1.5 rounded-lg bg-[#151F35] hover:bg-[#1B2942] border border-[#2B3A54] hover:border-[#43D9F5]/50 text-xs text-[#B4C0D4] hover:text-[#43D9F5] transition-all cursor-pointer"
              >
                {cq}
              </button>
            ))}
          </div>
        </div>

        {/* Message Thread */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex flex-col ${m.sender === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[90%] rounded-2xl px-5 py-3.5 text-sm leading-relaxed shadow-md ${
                  m.sender === 'user'
                    ? 'bg-[#818CF8]/20 border border-[#818CF8]/40 text-[#F1F5F9]'
                    : 'bg-[#151F35] border border-[#2B3A54] text-[#F1F5F9]'
                }`}
              >
                <div className="whitespace-pre-wrap font-sans">{m.text}</div>

                {/* Suggested Navigation Link */}
                {m.data?.recommended_navigation && (
                  <div className="mt-3.5 pt-3 border-t border-[#2B3A54] flex items-center justify-between">
                    <span className="text-xs text-[#B4C0D4] font-mono">Suggested Workspace:</span>
                    <button
                      onClick={() => handleNavigate(m.data!.recommended_navigation!.path)}
                      className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-[#43D9F5]/20 hover:bg-[#43D9F5]/30 border border-[#43D9F5]/40 text-[#43D9F5] text-xs font-semibold transition-all cursor-pointer"
                    >
                      <Compass className="w-3.5 h-3.5" />
                      <span>{m.data.recommended_navigation.label}</span>
                    </button>
                  </div>
                )}

                {/* Grounding Source Citations */}
                {m.data?.source_references && m.data.source_references.length > 0 && (
                  <div className="mt-3 pt-2 text-xs text-[#B4C0D4] font-mono flex items-center space-x-2 flex-wrap gap-y-1">
                    <div className="flex items-center space-x-1 text-[#43D9F5]">
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>Sources:</span>
                    </div>
                    {m.data.source_references.map((src, sIdx) => (
                      <span key={sIdx} className="bg-[#0B1020] px-2 py-0.5 rounded border border-[#2B3A54] text-[#F1F5F9]">
                        {src}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <span className="text-xs font-mono text-[#64748B] mt-1.5 px-2">{m.timestamp}</span>
            </div>
          ))}

          {loading && (
            <div className="flex items-center space-x-3 text-sm text-[#B4C0D4] bg-[#151F35] p-4 rounded-xl border border-[#2B3A54] w-fit">
              <Loader2 className="w-4 h-4 animate-spin text-[#43D9F5]" />
              <span>Synthesizing repository knowledge and active execution trace...</span>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-sm text-red-300 flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-[#2B3A54] bg-[#10182B]">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend(inputQuery);
            }}
            className="flex items-center space-x-2.5"
          >
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask about routing scores, models, NDT protocols..."
              className="flex-1 px-4 py-2.5 rounded-xl bg-[#0B1020] border border-[#2B3A54] text-sm text-[#F1F5F9] placeholder-[#64748B] focus:outline-none focus:border-[#43D9F5]"
            />
            <button
              type="submit"
              disabled={loading || !inputQuery.trim()}
              className="px-4 py-2.5 rounded-xl bg-[#43D9F5] hover:bg-[#43D9F5]/90 disabled:opacity-40 text-[#0B1020] font-bold text-sm transition-all flex items-center space-x-1.5 shadow-md cursor-pointer"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="mt-2.5 text-center text-xs text-[#64748B] font-mono">
            Anna University 23CSP701 • IEEE Access Spec Grounding • Zero Hallucination
          </div>
        </div>
      </div>
    </div>
  );
};
