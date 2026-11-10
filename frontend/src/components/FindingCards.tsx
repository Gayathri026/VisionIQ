import React from 'react';
import { Finding } from '../types';
import { ShieldAlert, AlertTriangle, CheckCircle, Search, HelpCircle, UserCheck } from 'lucide-react';

interface Props {
  findings: Finding[];
  selectedFindingUid?: string | null;
  onSelectFinding: (uid: string | null) => void;
}

export const FindingCards: React.FC<Props> = ({ findings, selectedFindingUid, onSelectFinding }) => {
  if (findings.length === 0) {
    return (
      <div className="tech-panel rounded-xl p-8 text-center text-graphite-400">
        <Search className="w-8 h-8 mx-auto mb-2 text-graphite-500" />
        <p className="text-xs">No discrete specialized findings extracted for this query.</p>
      </div>
    );
  }

  const getSourceBadge = (source: string) => {
    switch (source) {
      case 'ppe':
        return <span className="bg-accent-emerald/20 text-accent-emerald border border-accent-emerald/30 text-[10px] font-mono px-2 py-0.5 rounded uppercase">PPE Safety</span>;
      case 'defect':
        return <span className="bg-accent-crimson/20 text-red-400 border border-red-500/30 text-[10px] font-mono px-2 py-0.5 rounded uppercase">Defect Inspection</span>;
      case 'ocr':
        return <span className="bg-purple-900/40 text-purple-300 border border-purple-500/30 text-[10px] font-mono px-2 py-0.5 rounded uppercase">OCR Text</span>;
      case 'scene':
        return <span className="bg-cyan-900/30 text-cyan-300 border border-cyan-500/30 text-[10px] font-mono px-2 py-0.5 rounded uppercase">Scene Context</span>;
      default:
        return <span className="bg-accent-blue/20 text-blue-300 border border-blue-500/30 text-[10px] font-mono px-2 py-0.5 rounded uppercase">Object Detection</span>;
    }
  };

  return (
    <div className="space-y-3">
      {findings.map((f) => {
        const isSelected = selectedFindingUid === f.finding_uid;
        return (
          <div
            key={f.finding_uid}
            onClick={() => onSelectFinding(isSelected ? null : f.finding_uid)}
            className={`tech-panel-interactive rounded-xl p-4 cursor-pointer ${
              isSelected ? 'border-accent-cyan shadow-md shadow-accent-cyan/10 bg-graphite-800/90' : ''
            }`}
          >
            {/* Top row */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs font-bold text-white bg-graphite-800 px-2 py-0.5 rounded border border-graphite-700">
                  {f.finding_uid}
                </span>
                {getSourceBadge(f.agent_source)}
              </div>

              {/* Confidence / Metric */}
              <div className="text-right">
                {f.detection_confidence !== null && f.detection_confidence !== undefined && (
                  <span className="text-xs font-mono text-accent-cyan font-bold">
                    {(f.detection_confidence * 100).toFixed(1)}% Conf
                  </span>
                )}
                {f.anomaly_score !== null && f.anomaly_score !== undefined && (
                  <span className="text-xs font-mono text-accent-amber font-bold ml-2">
                    Anomaly {f.anomaly_score.toFixed(3)}
                  </span>
                )}
              </div>
            </div>

            {/* Finding Label */}
            <h4 className="text-sm font-semibold text-white mb-1.5">{f.label}</h4>

            {/* Observed Evidence */}
            <p className="text-xs text-graphite-300 mb-2 leading-relaxed font-sans bg-graphite-900/40 p-2.5 rounded border border-graphite-800/60">
              <span className="font-semibold text-graphite-400">Observed Evidence: </span>
              {f.observed_evidence}
            </p>

            {/* Root Cause & Recommendations if available */}
            {f.root_cause_hypothesis && (
              <div className="text-xs text-graphite-300 mb-1.5 font-sans">
                <span className="font-semibold text-accent-amber">Causal Hypothesis: </span>
                {f.root_cause_hypothesis}
              </div>
            )}

            {f.recommendation && (
              <div className="text-xs text-graphite-300 mb-2 font-sans">
                <span className="font-semibold text-accent-emerald">Corrective Action: </span>
                {f.recommendation}
              </div>
            )}

            {/* Bottom Footer Badges */}
            <div className="flex items-center justify-between pt-2 border-t border-graphite-800 text-[11px] font-mono text-graphite-400">
              {f.bbox && (
                <span>BBox: [{f.bbox.map((v) => Math.round(v)).join(', ')}]</span>
              )}

              {f.human_review_required ? (
                <span className="flex items-center space-x-1 text-accent-amber font-semibold">
                  <UserCheck className="w-3.5 h-3.5" />
                  <span>Human Inspection Required</span>
                </span>
              ) : (
                <span className="flex items-center space-x-1 text-accent-emerald">
                  <CheckCircle className="w-3.5 h-3.5" />
                  <span>Evidence Grounded</span>
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
