import React from 'react';
import { RAGReference } from '../types';
import { BookOpen, ExternalLink, Bookmark } from 'lucide-react';

interface Props {
  references: RAGReference[];
  evidenceSupportScore?: number;
}

export const RAGPanel: React.FC<Props> = ({ references, evidenceSupportScore }) => {
  if (references.length === 0) {
    return (
      <div className="tech-panel rounded-xl p-5 text-center text-graphite-400">
        <BookOpen className="w-6 h-6 mx-auto mb-2 text-graphite-500" />
        <p className="text-xs">No regulatory RAG citations required for this query.</p>
      </div>
    );
  }

  return (
    <div className="tech-panel rounded-xl p-5">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-graphite-700/80">
        <div className="flex items-center space-x-2">
          <Bookmark className="w-4 h-4 text-accent-cyan" />
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider font-mono">
            RAG Knowledge Grounding & Standards Citations
          </h3>
        </div>
        {evidenceSupportScore !== undefined && (
          <span className="text-xs font-mono text-accent-emerald bg-accent-emerald/10 border border-accent-emerald/30 px-2 py-0.5 rounded">
            Proven Score: {(evidenceSupportScore * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <div className="space-y-3">
        {references.map((ref, idx) => (
          <div key={idx} className="p-3.5 rounded-lg bg-graphite-800/40 border border-graphite-700/70">
            <div className="flex items-start justify-between mb-1.5">
              <div>
                <h4 className="text-xs font-semibold text-white">{ref.document_title}</h4>
                <span className="text-[11px] font-mono text-accent-cyan">{ref.section_reference}</span>
              </div>
              <span className="text-[11px] font-mono text-graphite-400">
                {(ref.similarity_score * 100).toFixed(1)}% Match
              </span>
            </div>

            <p className="text-xs text-graphite-300 italic mb-2 leading-relaxed bg-graphite-900/60 p-2.5 rounded border border-graphite-800">
              "{ref.excerpt}"
            </p>

            {ref.source_uri && (
              <a
                href={ref.source_uri}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center space-x-1 text-[11px] font-mono text-accent-cyan hover:underline"
              >
                <span>Official Documentation Source</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
