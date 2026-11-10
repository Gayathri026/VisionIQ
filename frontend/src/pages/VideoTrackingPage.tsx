import React, { useState } from 'react';
import { api } from '../services/api';
import { Video, Upload, Play, Clock, Users, Cpu, Loader2, AlertCircle } from 'lucide-react';

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

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleProcessVideo = async () => {
    if (!videoFile) {
      setError("Please select a video file first.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const data = await api.processVideo(videoFile, query, sampleFps, maxFrames);
      setVideoResult(data);
      setSelectedFrameIdx(0);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || "Video processing failed. Check video file encoding.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-6">
      <div className="mb-6">
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <Video className="w-5 h-5 text-accent-amber" />
          <span>Video Stream Tracking & Non-Duplicate Entity Analysis</span>
        </h2>
        <p className="text-xs text-graphite-400 font-mono">
          Memory-efficient frame sampling with IoU Multi-Object Tracking to prevent double-counting across temporal frames.
        </p>
      </div>

      {/* Video Control Card */}
      <div className="tech-panel rounded-xl p-5 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* File input */}
          <div className="md:col-span-4">
            <label className="block text-xs font-semibold text-graphite-300 uppercase tracking-wider font-mono mb-1.5">
              Select Video Stream (.mp4, .mov)
            </label>
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoSelect}
              className="hidden"
              id="video-upload"
            />
            <label
              htmlFor="video-upload"
              className="flex items-center justify-between w-full px-3.5 py-2.5 rounded-lg border border-graphite-700 bg-graphite-800/80 hover:bg-graphite-800 text-xs font-mono text-graphite-300 cursor-pointer transition-colors"
            >
              <div className="flex items-center space-x-2 truncate">
                <Upload className="w-4 h-4 text-accent-amber flex-shrink-0" />
                <span className="truncate">{videoFile ? videoFile.name : "Select video..."}</span>
              </div>
              <span className="text-[10px] bg-graphite-700 px-2 py-0.5 rounded text-white font-sans">Browse</span>
            </label>
          </div>

          {/* Sampling controls */}
          <div className="md:col-span-3">
            <label className="block text-xs font-semibold text-graphite-300 uppercase tracking-wider font-mono mb-1.5">
              Sampling Rate: <span className="text-accent-cyan font-bold">{sampleFps} FPS</span>
            </label>
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.5"
              value={sampleFps}
              onChange={(e) => setSampleFps(parseFloat(e.target.value))}
              className="w-full accent-accent-cyan cursor-pointer"
            />
          </div>

          <div className="md:col-span-3">
            <label className="block text-xs font-semibold text-graphite-300 uppercase tracking-wider font-mono mb-1.5">
              Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full px-3.5 py-2 rounded-lg border border-graphite-700 bg-graphite-800 text-xs text-white"
            />
          </div>

          <div className="md:col-span-2 flex items-end">
            <button
              onClick={handleProcessVideo}
              disabled={loading || !videoFile}
              className="w-full h-[40px] flex items-center justify-center space-x-2 bg-gradient-to-r from-accent-amber to-amber-600 hover:opacity-90 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow-md transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Tracking...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Process Video</span>
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-red-950/40 border border-red-500/40 text-xs text-red-300 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Main Video Presentation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Video Player & Temporal Timeline */}
        <div className="lg:col-span-7 space-y-6">
          <div className="tech-panel rounded-xl overflow-hidden p-4">
            <h3 className="text-xs font-mono uppercase tracking-wider text-graphite-300 mb-3 flex items-center space-x-2">
              <Play className="w-4 h-4 text-accent-amber" />
              <span>Video Playback & Frame Sampling</span>
            </h3>
            {videoPreview ? (
              <video
                src={videoPreview}
                controls
                className="w-full max-h-[380px] object-contain rounded-lg border border-graphite-800 bg-black"
              />
            ) : (
              <div className="h-64 flex items-center justify-center text-xs text-graphite-500 font-mono border border-dashed border-graphite-800 rounded-lg">
                Upload video above to activate temporal inspector
              </div>
            )}
          </div>

          {/* Temporal Timeline Scrubber */}
          {videoResult && videoResult.frame_results && (
            <div className="tech-panel rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono font-semibold text-white">Sampled Frames Timeline</span>
                <span className="text-xs font-mono text-accent-cyan">
                  Frame {selectedFrameIdx + 1} of {videoResult.frame_results.length}
                </span>
              </div>

              <div className="flex items-center space-x-2 overflow-x-auto pb-2">
                {videoResult.frame_results.map((fr: any, idx: number) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedFrameIdx(idx)}
                    className={`px-3 py-2 rounded-lg text-xs font-mono text-center flex-shrink-0 transition-all border ${
                      selectedFrameIdx === idx
                        ? 'bg-accent-amber text-white border-amber-400 font-bold'
                        : 'bg-graphite-800 text-graphite-300 border-graphite-700 hover:border-graphite-600'
                    }`}
                  >
                    <div>{fr.timestamp_sec}s</div>
                    <div className="text-[10px] opacity-80">{fr.findings_count} finds</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Unique Entity Tracking Metrics & Findings */}
        <div className="lg:col-span-5 space-y-6">
          {videoResult && (
            <div className="tech-panel rounded-xl p-5">
              <h3 className="text-xs font-mono uppercase tracking-wider text-accent-cyan font-bold mb-3 flex items-center space-x-1.5">
                <Users className="w-4 h-4" />
                <span>Multi-Object Tracking Summary (Non-Duplicate)</span>
              </h3>

              <div className="grid grid-cols-2 gap-3 mb-4 font-mono">
                <div className="p-3 rounded-lg bg-graphite-800/60 border border-graphite-700">
                  <div className="text-xs text-graphite-400">Duration</div>
                  <div className="text-lg font-bold text-white">{videoResult.duration_sec}s</div>
                </div>
                <div className="p-3 rounded-lg bg-graphite-800/60 border border-graphite-700">
                  <div className="text-xs text-graphite-400">Total Tracked</div>
                  <div className="text-lg font-bold text-accent-emerald">{videoResult.total_unique_entities} Unique</div>
                </div>
              </div>

              <div className="border-t border-graphite-800 pt-3">
                <span className="text-xs font-mono text-graphite-400 block mb-2">Tracked Entity Breakdown:</span>
                <div className="space-y-1.5 font-mono text-xs">
                  {Object.entries(videoResult.unique_tracked_entities || {}).map(([label, count]: any) => (
                    <div key={label} className="flex justify-between items-center py-1 px-2 rounded bg-graphite-800/40">
                      <span className="capitalize text-white">{label}</span>
                      <span className="text-accent-cyan font-bold">{count} unique count</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Findings on selected frame */}
          {videoResult && videoResult.frame_results && videoResult.frame_results[selectedFrameIdx] && (
            <div className="tech-panel rounded-xl p-5">
              <h4 className="text-xs font-mono uppercase text-white font-semibold mb-3">
                Findings at {videoResult.frame_results[selectedFrameIdx].timestamp_sec}s (Frame #{videoResult.frame_results[selectedFrameIdx].frame_index})
              </h4>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {videoResult.frame_results[selectedFrameIdx].findings.map((f: any, fIdx: number) => (
                  <div key={fIdx} className="p-2.5 rounded-lg bg-graphite-800/50 border border-graphite-700 text-xs">
                    <div className="flex justify-between font-mono mb-1">
                      <span className="text-white font-semibold">{f.label}</span>
                      <span className="text-accent-amber font-bold">Track #{f.track_id || 'N/A'}</span>
                    </div>
                    <p className="text-[11px] text-graphite-300">{f.observed_evidence}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
