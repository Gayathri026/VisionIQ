import React, { useState } from 'react';
import { Finding } from '../types';
import { Eye, Layers, Download, FileText, CheckCircle2 } from 'lucide-react';

interface Props {
  originalUrl?: string;
  annotatedUrl?: string;
  pdfUrl?: string;
  findings: Finding[];
  selectedFindingUid?: string | null;
  onSelectFinding: (uid: string | null) => void;
}

export const EvidenceWorkspace: React.FC<Props> = ({
  originalUrl,
  annotatedUrl,
  pdfUrl,
  findings,
  selectedFindingUid,
  onSelectFinding
}) => {
  const [viewMode, setViewMode] = useState<'annotated' | 'original' | 'split'>('annotated');

  if (!originalUrl && !annotatedUrl) {
    return (
      <div className="tech-panel rounded-xl p-12 text-center flex flex-col items-center justify-center border-dashed border-graphite-700">
        <Layers className="w-12 h-12 text-graphite-500 mb-3" />
        <h3 className="text-sm font-semibold text-graphite-300">No Visual Evidence Loaded</h3>
        <p className="text-xs text-graphite-500 max-w-sm mt-1">
          Upload an industrial inspection image or video stream to trigger dynamic multi-agent perception.
        </p>
      </div>
    );
  }

  return (
    <div className="tech-panel rounded-xl overflow-hidden flex flex-col h-full">
      {/* Workspace Controls */}
      <div className="px-5 py-3 border-b border-graphite-700/80 bg-graphite-800/50 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-semibold text-white uppercase tracking-wider font-mono flex items-center space-x-1.5">
            <Eye className="w-3.5 h-3.5 text-accent-cyan" />
            <span>Visual Evidence Workspace</span>
          </span>
          <span className="text-xs font-mono bg-graphite-700 text-graphite-300 px-2 py-0.5 rounded">
            {findings.length} Entities Identified
          </span>
        </div>

        {/* View mode toggles */}
        <div className="flex items-center space-x-2">
          <div className="bg-graphite-900 p-0.5 rounded-lg border border-graphite-700 flex text-xs font-mono">
            <button
              onClick={() => setViewMode('annotated')}
              className={`px-3 py-1 rounded transition-colors ${
                viewMode === 'annotated' ? 'bg-accent-blue text-white' : 'text-graphite-400 hover:text-white'
              }`}
            >
              Annotated
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-3 py-1 rounded transition-colors ${
                viewMode === 'original' ? 'bg-accent-blue text-white' : 'text-graphite-400 hover:text-white'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1 rounded transition-colors ${
                viewMode === 'split' ? 'bg-accent-blue text-white' : 'text-graphite-400 hover:text-white'
              }`}
            >
              Split View
            </button>
          </div>

          {pdfUrl && (
            <a
              href={pdfUrl}
              download
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1.5 bg-accent-blue hover:bg-accent-blue/90 text-white text-xs font-mono font-medium px-3 py-1.5 rounded-md transition-colors shadow-sm"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Download PDF</span>
            </a>
          )}
        </div>
      </div>

      {/* Visual Canvas Area */}
      <div className="p-4 bg-graphite-950 flex-1 flex items-center justify-center overflow-hidden min-h-[420px]">
        {viewMode === 'split' ? (
          <div className="grid grid-cols-2 gap-4 w-full h-full items-center">
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-mono text-graphite-400 mb-1.5">Source Input</span>
              <img
                src={originalUrl}
                alt="Source Image"
                className="max-h-[380px] w-auto object-contain rounded-lg border border-graphite-800"
              />
            </div>
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-mono text-accent-cyan mb-1.5">Annotated Evidence</span>
              <img
                src={annotatedUrl || originalUrl}
                alt="Annotated Image"
                className="max-h-[380px] w-auto object-contain rounded-lg border border-accent-blue/40 shadow-lg shadow-accent-blue/10"
              />
            </div>
          </div>
        ) : (
          <div className="relative flex items-center justify-center w-full h-full">
            <img
              src={viewMode === 'annotated' ? (annotatedUrl || originalUrl) : originalUrl}
              alt="Visual Evidence Canvas"
              className="max-h-[500px] w-auto object-contain rounded-lg border border-graphite-800 shadow-2xl"
            />
          </div>
        )}
      </div>

      {/* Footer Finding Pills */}
      {findings.length > 0 && (
        <div className="px-5 py-2.5 border-t border-graphite-800 bg-graphite-900/60 overflow-x-auto flex items-center space-x-2">
          <span className="text-[11px] font-mono text-graphite-400 whitespace-nowrap">Filter Entity:</span>
          {findings.map((f) => {
            const isSelected = selectedFindingUid === f.finding_uid;
            return (
              <button
                key={f.finding_uid}
                onClick={() => onSelectFinding(isSelected ? null : f.finding_uid)}
                className={`text-[11px] font-mono px-2.5 py-1 rounded-md transition-all whitespace-nowrap border ${
                  isSelected
                    ? 'bg-accent-blue text-white border-accent-cyan shadow-sm shadow-accent-blue/30'
                    : 'bg-graphite-800 text-graphite-300 border-graphite-700 hover:border-graphite-600'
                }`}
              >
                {f.finding_uid}: {f.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
