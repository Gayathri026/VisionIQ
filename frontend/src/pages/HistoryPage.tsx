import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { HistorySession } from '../types';
import { History, Search, FileText, CheckCircle2, Clock, Eye } from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [sessions, setSessions] = useState<HistorySession[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await api.getHistory(50);
        setSessions(data);
      } catch (e) {
        console.error("Failed to load history", e);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const filteredSessions = sessions.filter(
    (s) =>
      s.user_query.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (s.detected_intent && s.detected_intent.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <History className="w-5 h-5 text-accent-cyan" />
            <span>Inspection & Orchestration Trace History</span>
          </h2>
          <p className="text-xs text-graphite-400 font-mono">
            Persistent SQLite database log of prior queries, findings, routing scores, and downloadable PDF reports.
          </p>
        </div>

        {/* Search Input */}
        <div className="relative w-72">
          <Search className="w-4 h-4 text-graphite-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search query or intent..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg border border-graphite-700 bg-graphite-800 text-xs text-white placeholder-graphite-500 focus:outline-none focus:border-accent-cyan"
          />
        </div>
      </div>

      {loading ? (
        <div className="tech-panel rounded-xl p-12 text-center text-xs font-mono text-graphite-400">
          Loading historical sessions...
        </div>
      ) : filteredSessions.length === 0 ? (
        <div className="tech-panel rounded-xl p-12 text-center text-xs font-mono text-graphite-400">
          No historical inspection sessions found.
        </div>
      ) : (
        <div className="tech-panel rounded-xl overflow-hidden">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-graphite-700 bg-graphite-800/40 text-graphite-400">
                <th className="py-3 px-4">Session ID</th>
                <th className="py-3 px-4">Date / Time</th>
                <th className="py-3 px-4">User Query</th>
                <th className="py-3 px-4">Detected Intent</th>
                <th className="py-3 px-4 text-center">Latency</th>
                <th className="py-3 px-4 text-center">Status</th>
                <th className="py-3 px-4 text-right">PDF Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-800">
              {filteredSessions.map((s) => (
                <tr key={s.id} className="hover:bg-graphite-800/30 transition-colors">
                  <td className="py-3 px-4 text-accent-cyan font-bold">
                    {s.id.slice(0, 8)}...
                  </td>
                  <td className="py-3 px-4 text-graphite-300">
                    {new Date(s.created_at).toLocaleString()}
                  </td>
                  <td className="py-3 px-4 text-white font-sans max-w-xs truncate">
                    {s.user_query}
                  </td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 rounded bg-graphite-800 text-graphite-300 border border-graphite-700">
                      {s.detected_intent || 'General'}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center text-graphite-300">
                    {s.total_latency_ms.toFixed(1)} ms
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] bg-accent-emerald/20 text-accent-emerald border border-accent-emerald/30">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>{s.status}</span>
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <a
                      href={api.getPdfReportUrl(s.id)}
                      download
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center space-x-1 text-accent-cyan hover:underline font-bold"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>Download</span>
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
