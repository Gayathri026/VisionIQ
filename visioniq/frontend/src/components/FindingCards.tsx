import React, { useState } from 'react';
import { Finding, RAGReference } from '../types';
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle,
  Search,
  HelpCircle,
  UserCheck,
  Cpu,
  Info,
  Filter,
  CheckCircle2,
  BookOpen,
  ArrowRight,
  ShieldCheck,
  Eye,
  SlidersHorizontal,
  Clock,
  Sparkles,
  ExternalLink
} from 'lucide-react';

interface Props {
  findings: Finding[];
  selectedFindingUid?: string | null;
  onSelectFinding: (uid: string | null) => void;
  ragReferences?: RAGReference[];
  activeFrameTimestamp?: number | null;
}

export const FindingCards: React.FC<Props> = ({
  findings,
  selectedFindingUid,
  onSelectFinding,
  ragReferences = [],
  activeFrameTimestamp
}) => {
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [uncertaintyFilter, setUncertaintyFilter] = useState<string>('all');
  const [reviewedFindings, setReviewedFindings] = useState<Record<string, boolean>>({});

  const toggleReviewed = (uid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setReviewedFindings(prev => ({ ...prev, [uid]: !prev[uid] }));
  };

  if (findings.length === 0) {
    return (
      <div className="command-panel rounded-xl p-10 text-center text-[#8998AF]">
        <Search className="w-10 h-10 mx-auto mb-3 text-[#546E9B]" />
        <h3 className="text-base font-bold text-[#F1F5F9]">No Specific Inspection Findings</h3>
        <p className="text-sm text-[#8998AF] mt-1.5 max-w-md mx-auto">
          Execute an inspection query to detect workplace PPE compliance, metal flaws, or OCR markings.
        </p>
      </div>
    );
  }

  // Compute metrics for inspection summary
  const totalFindings = findings.length;
  const criticalFindings = findings.filter(f =>
    f.label.toLowerCase().includes('violation') ||
    f.label.toLowerCase().includes('no-') ||
    f.is_confirmed_defect === true
  ).length;
  const uncertainFindings = findings.filter(f =>
    f.human_review_required || f.uncertainty_flag || (f.detection_confidence && f.detection_confidence < 0.65)
  ).length;
  const confirmedFindings = totalFindings - uncertainFindings;

  // Filter findings
  const filteredFindings = findings.filter(f => {
    // Type Filter
    if (typeFilter !== 'all') {
      if (typeFilter === 'ppe' && f.agent_source !== 'ppe') return false;
      if (typeFilter === 'defect' && f.agent_source !== 'defect') return false;
      if (typeFilter === 'ocr' && f.agent_source !== 'ocr') return false;
      if (typeFilter === 'object' && f.agent_source !== 'object') return false;
    }

    // Severity Filter
    if (severityFilter !== 'all') {
      const isCritical = f.label.toLowerCase().includes('violation') || f.label.toLowerCase().includes('no-') || f.is_confirmed_defect === true;
      const isWarning = f.human_review_required || f.uncertainty_flag;
      if (severityFilter === 'critical' && !isCritical) return false;
      if (severityFilter === 'warning' && !isWarning) return false;
      if (severityFilter === 'info' && (isCritical || isWarning)) return false;
    }

    // Uncertainty Filter
    if (uncertaintyFilter !== 'all') {
      const isReview = f.human_review_required || f.uncertainty_flag;
      if (uncertaintyFilter === 'review_required' && !isReview) return false;
      if (uncertaintyFilter === 'confirmed' && isReview) return false;
    }

    return true;
  });

  const getSourceBadge = (finding: Finding) => {
    const isCritical = finding.label.toLowerCase().includes('violation') ||
                       finding.label.toLowerCase().includes('no-') ||
                       finding.is_confirmed_defect === true;

    if (finding.agent_source === 'ppe') {
      if (isCritical) {
        return (
          <span className="bg-[#FB7185]/20 text-[#FB7185] border border-[#FB7185]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase flex items-center space-x-1.5">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>PPE Violation</span>
          </span>
        );
      }
      return (
        <span className="bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase flex items-center space-x-1.5">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>PPE Verified</span>
        </span>
      );
    }

    if (finding.agent_source === 'defect') {
      if (finding.is_confirmed_defect === true) {
        return (
          <span className="bg-[#FB7185]/20 text-[#FB7185] border border-[#FB7185]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase flex items-center space-x-1.5">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Confirmed Defect</span>
          </span>
        );
      }
      return (
        <span className="bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase flex items-center space-x-1.5">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>Heuristic Anomaly</span>
        </span>
      );
    }

    if (finding.agent_source === 'ocr') {
      return (
        <span className="bg-[#818CF8]/20 text-[#818CF8] border border-[#818CF8]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase">
          OCR Text
        </span>
      );
    }

    return (
      <span className="bg-[#43D9F5]/20 text-[#43D9F5] border border-[#43D9F5]/40 text-xs font-mono font-bold px-2.5 py-1 rounded-md uppercase">
        Object
      </span>
    );
  };

  return (
    <div className="space-y-4">
      {/* 4.1 Compact Inspection Summary Metrics Ribbon */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="command-panel p-3.5 flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-[#8998AF] uppercase tracking-wider font-semibold">Total Detections</div>
            <div className="text-2xl font-extrabold text-[#F1F5F9] font-mono mt-0.5">{totalFindings}</div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#43D9F5]">
            <Eye className="w-4 h-4" />
          </div>
        </div>

        <div className="command-panel p-3.5 flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-[#8998AF] uppercase tracking-wider font-semibold">Confirmed</div>
            <div className="text-2xl font-extrabold text-[#34D399] font-mono mt-0.5">{confirmedFindings}</div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#34D399]">
            <CheckCircle className="w-4 h-4" />
          </div>
        </div>

        <div className="command-panel p-3.5 flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-[#8998AF] uppercase tracking-wider font-semibold">Requires Review</div>
            <div className="text-2xl font-extrabold text-[#FBBF24] font-mono mt-0.5">{uncertainFindings}</div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#FBBF24]">
            <AlertTriangle className="w-4 h-4" />
          </div>
        </div>

        <div className="command-panel p-3.5 flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-[#8998AF] uppercase tracking-wider font-semibold">Critical Findings</div>
            <div className="text-2xl font-extrabold text-[#FB7185] font-mono mt-0.5">{criticalFindings}</div>
          </div>
          <div className="w-9 h-9 rounded-lg bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#FB7185]">
            <ShieldAlert className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="command-panel p-3 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#F1F5F9]">
          <SlidersHorizontal className="w-4 h-4 text-[#43D9F5]" />
          <span>FINDING FILTERS:</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Category Filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="bg-[#10182B] border border-[#2B3A54] rounded-lg px-3 py-1.5 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#43D9F5]"
          >
            <option value="all">All Categories</option>
            <option value="ppe">PPE Compliance</option>
            <option value="defect">Surface Defects</option>
            <option value="ocr">OCR Text</option>
            <option value="object">General Objects</option>
          </select>

          {/* Severity Filter */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="bg-[#10182B] border border-[#2B3A54] rounded-lg px-3 py-1.5 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#43D9F5]"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical Only</option>
            <option value="warning">Warnings Only</option>
            <option value="info">Informational</option>
          </select>

          {/* Uncertainty Filter */}
          <select
            value={uncertaintyFilter}
            onChange={(e) => setUncertaintyFilter(e.target.value)}
            className="bg-[#10182B] border border-[#2B3A54] rounded-lg px-3 py-1.5 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#43D9F5]"
          >
            <option value="all">All Verification States</option>
            <option value="confirmed">Confirmed</option>
            <option value="review_required">Requires Human Review</option>
          </select>
        </div>
      </div>

      {/* 4.2 Finding Cards List */}
      <div className="space-y-3">
        {filteredFindings.map((finding) => {
          const isSelected = selectedFindingUid === finding.finding_uid;
          const isReviewed = reviewedFindings[finding.finding_uid] || false;
          const isCritical = finding.label.toLowerCase().includes('violation') ||
                             finding.label.toLowerCase().includes('no-') ||
                             finding.is_confirmed_defect === true;

          return (
            <div
              key={finding.finding_uid}
              onClick={() => onSelectFinding(isSelected ? null : finding.finding_uid)}
              className={`command-panel p-4 transition-all cursor-pointer border ${
                isSelected
                  ? 'border-[#43D9F5] bg-[#1B2942] shadow-md shadow-[#43D9F5]/10'
                  : 'hover:border-[#3A4D6F] bg-[#151F35]'
              }`}
            >
              {/* Card Header */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#2B3A54]">
                <div className="flex items-center space-x-2.5 flex-wrap">
                  <span className="text-[#43D9F5] font-extrabold text-sm font-mono tracking-tight">
                    {finding.finding_uid}
                  </span>
                  <span className="text-[#8998AF]">•</span>
                  <span className="text-[#F1F5F9] font-bold text-base">
                    {finding.label}
                  </span>
                  {getSourceBadge(finding)}
                  {isReviewed && (
                    <span className="bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40 text-[11px] font-mono px-2 py-0.5 rounded flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>Operator Reviewed</span>
                    </span>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => toggleReviewed(finding.finding_uid, e)}
                    className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                      isReviewed
                        ? 'bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40 hover:bg-[#34D399]/30'
                        : 'bg-[#10182B] text-[#8998AF] border border-[#2B3A54] hover:text-white'
                    }`}
                  >
                    {isReviewed ? '✓ Verified' : 'Mark Reviewed'}
                  </button>
                </div>
              </div>

              {/* Card Body: Confidence Disambiguation & Spatial Evidence */}
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {/* Left: Physical Observations & Explicit Confidences */}
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2 font-mono">
                    {/* 4.3 Explicit Score Separation */}
                    {finding.detection_confidence != null && (
                      <span className="bg-[#10182B] text-[#34D399] border border-[#34D399]/30 px-2 py-1 rounded font-semibold">
                        Detection Conf: {(finding.detection_confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {finding.anomaly_score != null && (
                      <span className="bg-[#10182B] text-[#FBBF24] border border-[#FBBF24]/30 px-2 py-1 rounded font-semibold">
                        Anomaly Score: {finding.anomaly_score.toFixed(3)}
                      </span>
                    )}
                    {finding.spatial_zone && (
                      <span className="bg-[#10182B] text-[#818CF8] border border-[#818CF8]/30 px-2 py-1 rounded font-semibold">
                        Anatomical Zone: {finding.spatial_zone}
                      </span>
                    )}
                    {finding.bbox && (
                      <span className="text-[#8998AF]">
                        BBox: [{finding.bbox.map(v => Math.round(v)).join(', ')}]
                      </span>
                    )}
                  </div>

                  {finding.observed_evidence && (
                    <div className="text-[#B4C0D4] leading-relaxed">
                      <strong className="text-[#F1F5F9] font-mono uppercase text-[11px] block text-[#43D9F5]">
                        Direct Visual Observation:
                      </strong>
                      {finding.observed_evidence}
                    </div>
                  )}

                  {finding.model_name && (
                    <div className="text-[11px] font-mono text-[#8998AF]">
                      Model: <span className="text-[#B4C0D4]">{finding.model_name}</span> ({finding.model_version || 'v1.0.0'})
                    </div>
                  )}
                </div>

                {/* Right: Hypotheses, CAPA & NDT Recommendations */}
                <div className="space-y-2 bg-[#10182B]/60 p-3 rounded-lg border border-[#2B3A54]">
                  {finding.root_cause_hypothesis && (
                    <div>
                      <span className="text-[11px] font-mono uppercase text-[#FBBF24] font-bold block">
                        Causal Hypothesis (Requires Physical Verification):
                      </span>
                      <p className="text-[#B4C0D4] text-xs mt-0.5 leading-relaxed">
                        {finding.root_cause_hypothesis}
                      </p>
                    </div>
                  )}

                  {finding.recommendation && (
                    <div>
                      <span className="text-[11px] font-mono uppercase text-[#34D399] font-bold block">
                        Corrective Recommendation & Protocol:
                      </span>
                      <p className="text-[#F1F5F9] text-xs mt-0.5 leading-relaxed font-medium">
                        {finding.recommendation}
                      </p>
                    </div>
                  )}

                  {finding.verification_procedure && (
                    <div className="text-[11px] font-mono text-[#8998AF] border-t border-[#2B3A54] pt-1.5 mt-1">
                      <strong className="text-[#43D9F5]">NDT Standard:</strong> {finding.verification_procedure}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
