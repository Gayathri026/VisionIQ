import React, { useState, useRef } from 'react';
import { api } from '../services/api';
import { Video, Upload, Play, Clock, Users, Cpu, Loader2, AlertCircle, ShieldAlert, CheckCircle2, Eye, XCircle, Film, Sparkles } from 'lucide-react';

export const VideoTrackingPage: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [query, setQuery] = useState("Track workers entering the active machinery zone and verify PPE compliance");
  const [sampleFps, setSampleFps] = useState(1.0);
  const [maxFrames, setMaxFrames] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videoResult, setVideoResult] = useState<any | null>(null);
  const [selectedFrameIdx, setSelectedFrameIdx] = useState<number>(0);
  const [viewMode, setViewMode] = useState<'annotated' | 'video'>('annotated');

  const abortControllerRef = useRef<AbortController | null>(null);

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setError(null);
      setVideoResult(null);
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    setError("Video tracking cancelled by user.");
  };

  const handleProcessVideo = async () => {
    if (!videoFile) {
      setError("Please select a video file first.");
      return;
    }
    setLoading(true);
    setError(null);

    abortControllerRef.current = new AbortController();

    try {
      const data = await api.processVideo(videoFile, query, sampleFps, maxFrames);
      setVideoResult(data);
      setSelectedFrameIdx(0);
      setViewMode('annotated');
    } catch (err: any) {
      if (err.name === 'CanceledError' || err.message === 'canceled') {
        setError("Processing cancelled.");
      } else {
        console.error(err);
        setError(err?.response?.data?.detail || "Video processing failed. Check video file encoding.");
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const currentFrame = videoResult?.frame_results?.[selectedFrameIdx];

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#FBBF24]/15 border border-[#FBBF24]/30 flex items-center justify-center text-[#FBBF24]">
            <Film className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              Spatio-Temporal Video Tracking & Incremental Inspection
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Incremental OpenCV Frame Extraction • IoU Multi-Object Tracking • Temporal Violation Persistence
            </p>
          </div>
        </div>
      </div>

      {/* Video Control Bar */}
      <div className="command-panel p-5">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* File input */}
          <div className="md:col-span-3">
            <label className="block text-xs font-semibold text-[#8998AF] uppercase tracking-wider font-mono mb-1.5">
              Select Video (.mp4, .mov, max 100MB)
            </label>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
              onChange={handleVideoSelect}
              className="hidden"
              id="video-upload"
            />
            <label
              htmlFor="video-upload"
              className="flex items-center justify-between w-full px-4 py-2.5 rounded-lg border border-[#2B3A54] bg-[#10182B] hover:bg-[#1B2942] text-xs font-mono text-[#B4C0D4] cursor-pointer transition-colors"
            >
              <div className="flex items-center space-x-2 truncate">
                <Upload className="w-4 h-4 text-[#FBBF24] flex-shrink-0" />
                <span className="truncate">{videoFile ? videoFile.name : "Select video..."}</span>
              </div>
              <span className="text-[11px] bg-[#151F35] border border-[#2B3A54] px-2 py-0.5 rounded text-white font-sans">Browse</span>
            </label>
          </div>

          {/* Sampling controls */}
          <div className="md:col-span-2">
            <label className="block text-xs font-semibold text-[#8998AF] uppercase tracking-wider font-mono mb-1.5">
              Rate: <span className="text-[#43D9F5] font-bold">{sampleFps} FPS</span>
            </label>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.5"
              value={sampleFps}
              onChange={(e) => setSampleFps(parseFloat(e.target.value))}
              className="w-full accent-[#43D9F5] cursor-pointer h-1.5 bg-[#2B3A54] rounded-lg"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-xs font-semibold text-[#8998AF] uppercase tracking-wider font-mono mb-1.5">
              Max Frames: <span className="text-[#34D399] font-bold">{maxFrames}</span>
            </label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={maxFrames}
              onChange={(e) => setMaxFrames(parseInt(e.target.value))}
              className="w-full accent-[#34D399] cursor-pointer h-1.5 bg-[#2B3A54] rounded-lg"
            />
          </div>

          <div className="md:col-span-3">
            <label className="block text-xs font-semibold text-[#8998AF] uppercase tracking-wider font-mono mb-1.5">
              Inspection Objective / Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white placeholder-[#8998AF] focus:outline-none focus:border-[#43D9F5]"
            />
          </div>

          <div className="md:col-span-2 flex items-end space-x-2">
            {!loading ? (
              <button
                onClick={handleProcessVideo}
                disabled={!videoFile}
                className="w-full h-[42px] flex items-center justify-center space-x-2 bg-gradient-to-r from-[#D97706] to-[#FBBF24] hover:opacity-90 disabled:opacity-50 text-[#0B1020] text-xs font-bold rounded-lg shadow-md transition-all font-mono"
              >
                <Play className="w-4 h-4 text-[#0B1020]" />
                <span>Track Video</span>
              </button>
            ) : (
              <button
                onClick={handleCancel}
                className="w-full h-[42px] flex items-center justify-center space-x-2 bg-[#FB7185]/20 border border-[#FB7185]/50 hover:bg-[#FB7185]/30 text-[#FB7185] text-xs font-bold rounded-lg shadow-md transition-all font-mono"
              >
                <XCircle className="w-4 h-4" />
                <span>Cancel Processing</span>
              </button>
            )}
          </div>
        </div>

        {loading && (
          <div className="mt-4 p-3 rounded-lg bg-[#43D9F5]/10 border border-[#43D9F5]/30 flex items-center space-x-3 text-xs text-[#43D9F5] font-mono">
            <Loader2 className="w-4 h-4 animate-spin text-[#43D9F5]" />
            <span>Incrementally decoding frames & executing multi-object tracking pipeline (memory safe)...</span>
          </div>
        )}

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-[#FB7185]/15 border border-[#FB7185]/40 text-xs text-[#FB7185] flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-[#FB7185] flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Main Video Presentation Grid (Image/Video Sized to Content, Independent of Right Side) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Viewport Sized to Video Content (self-start + sticky) */}
        <div className="lg:col-span-7 space-y-4 self-start lg:sticky lg:top-20">
          <div className="command-panel p-4">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-[#2B3A54]">
              <h3 className="text-xs font-mono uppercase tracking-wider text-[#F1F5F9] font-bold flex items-center space-x-2">
                <Play className="w-4 h-4 text-[#FBBF24]" />
                <span>Frame Inspector Canvas</span>
              </h3>
              {videoResult && (
                <div className="flex items-center space-x-1 bg-[#10182B] rounded-lg p-0.5 border border-[#2B3A54] text-xs font-mono">
                  <button
                    onClick={() => setViewMode('annotated')}
                    className={`px-3 py-1 rounded-md transition-colors ${
                      viewMode === 'annotated' ? 'bg-[#1B2942] text-white font-bold border border-[#43D9F5]/40' : 'text-[#8998AF] hover:text-white'
                    }`}
                  >
                    Annotated Overlays
                  </button>
                  <button
                    onClick={() => setViewMode('video')}
                    className={`px-3 py-1 rounded-md transition-colors ${
                      viewMode === 'video' ? 'bg-[#1B2942] text-white font-bold border border-[#43D9F5]/40' : 'text-[#8998AF] hover:text-white'
                    }`}
                  >
                    Original Video
                  </button>
                </div>
              )}
            </div>

            {viewMode === 'annotated' && currentFrame?.annotated_frame_url ? (
              <div className="relative flex items-center justify-center bg-[#070A14] rounded-lg p-2">
                <img
                  src={currentFrame.annotated_frame_url}
                  alt={`Annotated Frame ${selectedFrameIdx + 1}`}
                  className="max-h-[min(65vh,520px)] w-auto max-w-full h-auto object-contain rounded-lg border border-[#2B3A54] shadow-2xl block"
                />
                <div className="absolute top-4 left-4 bg-[#0B1020]/80 backdrop-blur px-3 py-1.5 rounded-lg text-xs font-mono text-white border border-[#2B3A54]">
                  Frame #{currentFrame.frame_index} | t={currentFrame.timestamp_sec}s
                </div>
              </div>
            ) : videoPreview ? (
              <div className="bg-[#070A14] rounded-lg p-2 flex items-center justify-center">
                <video
                  src={videoPreview}
                  controls
                  className="max-h-[min(65vh,520px)] w-auto max-w-full h-auto object-contain rounded-lg border border-[#2B3A54] block"
                />
              </div>
            ) : (
              <div className="h-56 flex items-center justify-center text-sm text-[#8998AF] font-mono border border-dashed border-[#2B3A54] rounded-lg">
                Upload video file above to activate temporal tracking
              </div>
            )}
          </div>

          {/* Temporal Timeline Scrubber */}
          {videoResult && videoResult.frame_results && (
            <div className="command-panel p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono font-semibold text-white">Sampled Frames Timeline</span>
                <span className="text-xs font-mono text-[#43D9F5]">
                  Frame {selectedFrameIdx + 1} of {videoResult.frame_results.length} (t={currentFrame?.timestamp_sec}s)
                </span>
              </div>

              <div className="flex items-center space-x-2.5 overflow-x-auto pb-2">
                {videoResult.frame_results.map((fr: any, idx: number) => {
                  const hasViolation = fr.findings?.some((f: any) =>
                    f.label?.includes('Violation') || f.observed_evidence?.includes('NO-')
                  );
                  return (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedFrameIdx(idx);
                        setViewMode('annotated');
                      }}
                      className={`px-3.5 py-2.5 rounded-lg text-xs font-mono text-center flex-shrink-0 transition-all border ${
                        selectedFrameIdx === idx
                          ? 'bg-[#1B2942] text-white border-[#43D9F5] font-bold shadow-lg shadow-[#43D9F5]/20'
                          : hasViolation
                          ? 'bg-[#FB7185]/15 text-[#FB7185] border-[#FB7185]/40 hover:border-[#FB7185]'
                          : 'bg-[#10182B] text-[#B4C0D4] border-[#2B3A54] hover:border-[#3A4D6F]'
                      }`}
                    >
                      <div className="flex items-center justify-center space-x-1.5">
                        <span>{fr.timestamp_sec}s</span>
                        {hasViolation && <span className="w-2 h-2 rounded-full bg-[#FB7185]" />}
                      </div>
                      <div className="text-[10px] text-[#8998AF] mt-0.5">{fr.findings_count} entities</div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Temporal Violation Persistence Analysis Table */}
          {videoResult?.temporal_violations && videoResult.temporal_violations.length > 0 && (
            <div className="command-panel p-5">
              <h4 className="text-xs font-mono uppercase tracking-wider text-[#FBBF24] font-bold mb-3 flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
                <ShieldAlert className="w-4 h-4 text-[#FBBF24]" />
                <span>Temporal Violation Persistence Reasoning</span>
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#2B3A54] text-[#8998AF]">
                      <th className="py-2.5 px-3">Track ID</th>
                      <th className="py-2.5 px-3">Entity</th>
                      <th className="py-2.5 px-3">Observed Span</th>
                      <th className="py-2.5 px-3">Violations</th>
                      <th className="py-2.5 px-3">Persistence Category</th>
                      <th className="py-2.5 px-3">Audit Protocol</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#2B3A54]/60">
                    {videoResult.temporal_violations.map((tv: any, idx: number) => (
                      <tr key={idx} className="hover:bg-[#1B2942]/40">
                        <td className="py-2.5 px-3 font-bold text-[#43D9F5]">Track #{tv.track_id}</td>
                        <td className="py-2.5 px-3 text-white capitalize">{tv.class_label}</td>
                        <td className="py-2.5 px-3 text-[#B4C0D4]">{tv.frames_observed} frames ({tv.duration_sec}s)</td>
                        <td className="py-2.5 px-3 text-[#FBBF24] font-bold">{tv.violation_frames} / {tv.frames_observed}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                            tv.is_persistent
                              ? 'bg-[#FB7185]/20 text-[#FB7185] border border-[#FB7185]/40'
                              : 'bg-[#FBBF24]/20 text-[#FBBF24] border border-[#FBBF24]/40'
                          }`}>
                            {tv.persistence_category}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-[11px] text-[#8998AF] max-w-xs">{tv.evidence_note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right: Unique Entity Tracking Metrics & Findings */}
        <div className="lg:col-span-5 space-y-5">
          {videoResult && (
            <div className="command-panel p-5 space-y-4">
              <h3 className="text-xs font-mono uppercase tracking-wider text-[#43D9F5] font-bold flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
                <Users className="w-4 h-4" />
                <span>Multi-Object Tracking (Non-Duplicate Counts)</span>
              </h3>

              {/* Per-Frame vs Unique Tracked Count Comparison */}
              <div className="grid grid-cols-2 gap-3 font-mono">
                <div className="p-3.5 rounded-lg bg-[#10182B] border border-[#2B3A54]">
                  <div className="text-xs text-[#8998AF]">Total Unique Tracked</div>
                  <div className="text-2xl font-extrabold text-[#34D399] mt-1">{videoResult.total_unique_entities} Entities</div>
                  <div className="text-[11px] text-[#8998AF] mt-0.5">Non-duplicate spatial tracks</div>
                </div>
                <div className="p-3.5 rounded-lg bg-[#10182B] border border-[#2B3A54]">
                  <div className="text-xs text-[#8998AF]">Frame Instantaneous</div>
                  <div className="text-2xl font-extrabold text-[#FBBF24] mt-1">
                    {currentFrame ? Number((Object.values(currentFrame.per_frame_counts || {}) as number[]).reduce((a: number, b: number) => a + b, 0)) : 0} Count
                  </div>
                  <div className="text-[11px] text-[#8998AF] mt-0.5">At t={currentFrame?.timestamp_sec}s</div>
                </div>
              </div>

              <div className="border-t border-[#2B3A54] pt-3 font-mono">
                <span className="text-xs text-[#8998AF] block mb-2 font-semibold">Unique Entity Tracks:</span>
                <div className="space-y-2 text-xs">
                  {Object.entries(videoResult.unique_tracked_entities || {}).map(([label, count]: any) => (
                    <div key={label} className="flex justify-between items-center py-1.5 px-3 rounded-lg bg-[#10182B] border border-[#2B3A54]">
                      <span className="capitalize text-white font-medium">{label}</span>
                      <span className="text-[#43D9F5] font-bold">{count} unique persistent track{count > 1 ? 's' : ''}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs font-mono text-[#B4C0D4]">
                <span className="text-[#34D399] font-bold">Synthesis:</span> {videoResult.synthesis_text}
              </div>
            </div>
          )}

          {/* Findings on selected frame */}
          {currentFrame && (
            <div className="command-panel p-5 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-[#2B3A54] font-mono">
                <h4 className="text-xs uppercase text-white font-semibold">
                  Frame #{currentFrame.frame_index} Evidence (t={currentFrame.timestamp_sec}s)
                </h4>
                <span className="text-xs text-[#43D9F5] font-bold">{currentFrame.findings_count} findings</span>
              </div>

              {currentFrame.findings?.length === 0 ? (
                <p className="text-xs text-[#8998AF] font-mono py-8 text-center">No entities detected in this frame.</p>
              ) : (
                <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
                  {currentFrame.findings.map((f: any, fIdx: number) => (
                    <div key={fIdx} className="p-3.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs space-y-1.5">
                      <div className="flex justify-between font-mono">
                        <span className="text-white font-bold text-sm">{f.label}</span>
                        <span className="text-[#FBBF24] font-mono font-bold">
                          {f.track_id ? `Track #${f.track_id}` : 'Instantaneous'}
                        </span>
                      </div>
                      <p className="text-xs text-[#B4C0D4] leading-relaxed">{f.observed_evidence}</p>
                      {f.bbox && (
                        <div className="text-[11px] font-mono text-[#8998AF]">
                          BBox: [{f.bbox.map((v: number) => v.toFixed(0)).join(', ')}]
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
