import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { ResearchExperimentsData, ExperimentComparison } from '../types';
import {
  FlaskConical,
  BarChart3,
  Clock,
  CheckCircle2,
  AlertCircle,
  TrendingDown,
  Layers,
  Database,
  Cpu,
  Info,
  ExternalLink,
  ShieldCheck,
  Loader2,
  Sparkles
} from 'lucide-react';

export const ResearchExperimentsPage: React.FC = () => {
  const [data, setData] = useState<ResearchExperimentsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        setLoading(true);
        const res = await api.getExperiments();
        setData(res);
      } catch (err: any) {
        console.error(err);
        setError("Failed to load research experiments from backend.");
      } finally {
        setLoading(false);
      }
    };
    fetchExperiments();
  }, []);

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#43D9F5]/15 border border-[#43D9F5]/30 flex items-center justify-center text-[#43D9F5]">
            <FlaskConical className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              Research Experiments & Pipeline Ablation Dashboard
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Empirical Comparisons: Fixed (all-agents) vs Rule-Based vs VisionIQ Adaptive (IEEE Access Track)
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2.5 text-xs font-mono">
          <span className="px-3.5 py-1.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-[#B4C0D4]">
            Reproducible Seed: <strong className="text-white">42</strong>
          </span>
          <span className="px-3.5 py-1.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-[#B4C0D4]">
            Target Venue: <strong className="text-[#43D9F5]">IEEE Access</strong>
          </span>
        </div>
      </div>

      {loading ? (
        <div className="command-panel p-16 text-center text-[#8998AF]">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-3 text-[#43D9F5]" />
          <span className="text-xs font-mono">Querying experiment database & SQLite records...</span>
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl bg-[#FB7185]/15 border border-[#FB7185]/40 text-xs text-[#FB7185] flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-[#FB7185]" />
          <span>{error}</span>
        </div>
      ) : (
        <>
          {/* Scientific Comparison Matrix: Fixed vs Rule-based vs Adaptive */}
          <div className="command-panel p-6 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-[#2B3A54]">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-[#43D9F5]" />
                  <span>Pipeline Architecture Comparison Benchmark Matrix</span>
                </h2>
                <p className="text-xs text-[#8998AF] font-sans mt-0.5">
                  Evaluated on 24 held-out test cases across 6 distinct capture sessions (zero near-duplicate data leakage).
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#2B3A54] text-[#8998AF]">
                    <th className="pb-3 px-2 font-medium">Pipeline Configuration</th>
                    <th className="pb-3 px-2 font-medium text-center">Status</th>
                    <th className="pb-3 px-2 font-medium text-center">Routing Precision</th>
                    <th className="pb-3 px-2 font-medium text-center">Routing Recall</th>
                    <th className="pb-3 px-2 font-medium text-center">Routing F1</th>
                    <th className="pb-3 px-2 font-medium text-center">Mean Agents</th>
                    <th className="pb-3 px-2 font-medium text-center">Median Latency</th>
                    <th className="pb-3 px-2 font-medium text-right text-[#34D399]">Latency Delta</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2B3A54]">
                  {data?.pipeline_comparisons.map((item, idx) => {
                    const isCompleted = item.status === 'completed';
                    const m = item.metrics;

                    return (
                      <tr key={idx} className="hover:bg-[#1B2942]/30 transition-colors">
                        <td className="py-3.5 px-2 pr-4">
                          <div className="font-semibold text-white font-sans text-xs flex items-center space-x-2">
                            <span>{item.name}</span>
                            {item.pipeline_type === 'adaptive' && (
                              <span className="bg-[#43D9F5]/20 text-[#43D9F5] border border-[#43D9F5]/40 px-2 py-0.5 rounded text-[10px] font-mono font-bold">
                                VisionIQ Proposed
                              </span>
                            )}
                          </div>
                          <p className="text-[11px] text-[#8998AF] font-sans mt-0.5">{item.description}</p>
                        </td>

                        <td className="py-3.5 px-2 text-center">
                          {isCompleted ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/30 font-bold">
                              COMPLETED
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#10182B] text-[#8998AF] border border-[#2B3A54]">
                              QUEUED
                            </span>
                          )}
                        </td>

                        <td className="py-3.5 px-2 text-center text-[#43D9F5]">
                          {isCompleted && m?.routing_precision !== undefined ? `${(m.routing_precision * 100).toFixed(1)}%` : '—'}
                        </td>

                        <td className="py-3.5 px-2 text-center text-[#43D9F5]">
                          {isCompleted && m?.routing_recall !== undefined ? `${(m.routing_recall * 100).toFixed(1)}%` : '—'}
                        </td>

                        <td className="py-3.5 px-2 text-center text-white font-bold">
                          {isCompleted && m?.routing_f1 !== undefined ? m.routing_f1.toFixed(3) : '—'}
                        </td>

                        <td className="py-3.5 px-2 text-center text-[#FBBF24]">
                          {isCompleted && m?.mean_agents_invoked !== undefined
                            ? m.mean_agents_invoked.toFixed(1)
                            : item.theoretical_mean_agents
                            ? `${item.theoretical_mean_agents.toFixed(1)} (theor)`
                            : '—'}
                        </td>

                        <td className="py-3.5 px-2 text-center text-[#F1F5F9]">
                          {isCompleted && m?.median_latency_ms !== undefined
                            ? `${m.median_latency_ms.toFixed(1)} ms`
                            : item.theoretical_latency_ms
                            ? `~${item.theoretical_latency_ms.toFixed(0)} ms`
                            : '—'}
                        </td>

                        <td className="py-3.5 px-2 text-right font-bold text-[#34D399]">
                          {isCompleted && m?.latency_reduction_pct !== undefined ? `+${m.latency_reduction_pct.toFixed(1)}%` : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Registry & Dataset Manifests Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Active Model Registry (6 Cols) */}
            <div className="lg:col-span-6 command-panel p-6 space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
                <Cpu className="w-4 h-4 text-[#43D9F5]" />
                <span>Registered Neural Model Weights</span>
              </h3>

              <div className="space-y-3">
                {data?.active_models.map((mod: any) => (
                  <div key={mod.id} className="p-3.5 rounded-xl bg-[#10182B] border border-[#2B3A54] space-y-1.5 text-xs font-mono">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white text-sm">{mod.model_name}</span>
                      <span className="text-[#43D9F5] bg-[#151F35] border border-[#43D9F5]/30 px-2 py-0.5 rounded text-[10px] font-bold">
                        {mod.version}
                      </span>
                    </div>
                    <div className="text-[#8998AF]">
                      Task: <strong className="text-white">{mod.task_type}</strong> • Target: <strong className="text-white">{mod.device_target}</strong>
                    </div>
                    {mod.val_accuracy !== null && (
                      <div className="text-[#34D399] font-semibold">
                        Validation Accuracy: {(mod.val_accuracy * 100).toFixed(1)}%
                      </div>
                    )}
                    <div className="text-[11px] text-[#8998AF] truncate pt-1 border-t border-[#2B3A54]">
                      Weights Path: {mod.weights_path}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Datasets Manifest (6 Cols) */}
            <div className="lg:col-span-6 command-panel p-6 space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
                <Database className="w-4 h-4 text-[#34D399]" />
                <span>Verified Scientific Benchmarks & Datasets</span>
              </h3>

              <div className="space-y-3">
                {data?.datasets.map((ds, idx) => (
                  <div key={idx} className="p-3.5 rounded-xl bg-[#10182B] border border-[#2B3A54] space-y-1.5 text-xs font-mono">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white text-sm font-sans">{ds.name}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/30 font-bold">
                        {ds.license}
                      </span>
                    </div>
                    <div className="text-[#8998AF]">
                      Task: <strong className="text-white">{ds.task}</strong> • Version: <strong className="text-white">{ds.version}</strong>
                    </div>
                    <div className="text-xs text-[#B4C0D4] font-sans flex flex-wrap gap-1.5 mt-1.5">
                      {ds.classes.map((cls, cIdx) => (
                        <span key={cIdx} className="bg-[#151F35] px-2 py-0.5 rounded text-[11px] text-[#B4C0D4] border border-[#2B3A54]">
                          {cls}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
