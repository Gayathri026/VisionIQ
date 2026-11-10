import React, { useState } from 'react';
import { AgentTrace } from '../types';
import { Calculator, ChevronDown, ChevronUp, Clock, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';

interface Props {
  traces: AgentTrace[];
  detectedIntent?: string;
  totalLatencyMs?: number;
}

export const OrchestrationTracePanel: React.FC<Props> = ({ traces, detectedIntent, totalLatencyMs }) => {
  const [expanded, setExpanded] = useState(true);

  const getDecisionBadge = (decision: string) => {
    switch (decision) {
      case 'SELECTED':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-accent-emerald/20 text-green-400 border border-green-500/30">
            <CheckCircle2 className="w-3 h-3" />
            <span>SELECTED</span>
          </span>
        );
      case 'MANDATORY_DEPENDENCY':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-accent-blue/20 text-blue-300 border border-blue-500/30">
            <AlertCircle className="w-3 h-3" />
            <span>DEPENDENCY</span>
          </span>
        );
      case 'FALLBACK':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-accent-amber/20 text-amber-300 border border-amber-500/30">
            <AlertCircle className="w-3 h-3" />
            <span>FALLBACK</span>
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-graphite-800 text-graphite-400 border border-graphite-700">
            <XCircle className="w-3 h-3" />
            <span>SKIPPED</span>
          </span>
        );
    }
  };

  return (
    <div className="tech-panel rounded-xl overflow-hidden mb-6">
      {/* Header */}
      <div
        className="px-5 py-3.5 flex items-center justify-between cursor-pointer border-b border-graphite-700/60 bg-graphite-800/40 hover:bg-graphite-800/70 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded-md bg-accent-blue/20 text-accent-cyan border border-accent-blue/30">
            <Calculator className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Cost-Aware Dynamic Orchestration Trace</h3>
            <p className="text-xs text-graphite-400 font-mono">
              Equation 1: S(a,q) = αR(a,q) + δE(a) − βC(a) − γL(a) | Intent: <span className="text-accent-cyan">{detectedIntent || 'General'}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {totalLatencyMs !== undefined && (
            <div className="flex items-center space-x-1.5 text-xs font-mono text-graphite-300 bg-graphite-800 px-2.5 py-1 rounded border border-graphite-700">
              <Clock className="w-3.5 h-3.5 text-accent-cyan" />
              <span>{totalLatencyMs.toFixed(1)} ms</span>
            </div>
          )}
          {expanded ? <ChevronUp className="w-4 h-4 text-graphite-400" /> : <ChevronDown className="w-4 h-4 text-graphite-400" />}
        </div>
      </div>

      {/* Body Table */}
      {expanded && (
        <div className="p-5 overflow-x-auto">
          {/* Equation Legend */}
          <div className="mb-4 p-3 rounded-lg bg-graphite-800/60 border border-graphite-700/80 text-xs font-mono grid grid-cols-2 md:grid-cols-5 gap-2 text-graphite-300">
            <div><span className="text-accent-cyan font-bold">α (Relevance):</span> 0.50</div>
            <div><span className="text-accent-emerald font-bold">δ (Capability):</span> 0.15</div>
            <div><span className="text-accent-amber font-bold">β (Cost Penalty):</span> 0.20</div>
            <div><span className="text-accent-crimson font-bold">γ (Latency Penalty):</span> 0.15</div>
            <div><span className="text-white font-bold">Threshold τ:</span> 0.45</div>
          </div>

          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-graphite-700 text-graphite-400">
                <th className="pb-2.5 font-medium">Agent</th>
                <th className="pb-2.5 font-medium text-center">Relevance R(a,q)</th>
                <th className="pb-2.5 font-medium text-center">Capability E(a)</th>
                <th className="pb-2.5 font-medium text-center">Cost C(a)</th>
                <th className="pb-2.5 font-medium text-center">Latency L(a)</th>
                <th className="pb-2.5 font-medium text-center text-white">Score S(a,q)</th>
                <th className="pb-2.5 font-medium text-center">Routing Decision</th>
                <th className="pb-2.5 font-medium text-right">Actual Runtime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-800">
              {traces.map((trace) => {
                const isSelected = trace.decision !== 'SKIPPED';
                return (
                  <tr key={trace.agent_name} className={`hover:bg-graphite-800/40 transition-colors ${isSelected ? 'bg-graphite-800/10' : 'opacity-60'}`}>
                    <td className="py-2.5 font-semibold text-white capitalize flex items-center space-x-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${isSelected ? 'bg-accent-cyan' : 'bg-graphite-600'}`} />
                      <span>{trace.agent_name.replace('_', ' ')}</span>
                    </td>
                    <td className="py-2.5 text-center text-accent-cyan">{trace.relevance_score.toFixed(2)}</td>
                    <td className="py-2.5 text-center text-accent-emerald">{trace.evidence_capability.toFixed(2)}</td>
                    <td className="py-2.5 text-center text-accent-amber">{trace.normalized_cost.toFixed(2)}</td>
                    <td className="py-2.5 text-center text-accent-crimson">{trace.expected_latency.toFixed(2)}</td>
                    <td className="py-2.5 text-center font-bold text-white">
                      {trace.final_routing_score.toFixed(3)}
                    </td>
                    <td className="py-2.5 text-center">{getDecisionBadge(trace.decision)}</td>
                    <td className="py-2.5 text-right text-graphite-300 font-mono">
                      {trace.actual_latency_ms > 0 ? `${trace.actual_latency_ms.toFixed(1)} ms` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
