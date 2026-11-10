import React, { useState, useRef, useEffect } from 'react';
import { api } from '../services/api';
import { Camera, Play, Pause, Square, AlertCircle, Cpu, Clock, RefreshCw } from 'lucide-react';

export const LiveCameraPage: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [query, setQuery] = useState("Continuous PPE safety compliance monitoring");
  const [error, setError] = useState<string | null>(null);
  const [liveFindings, setLiveFindings] = useState<any[]>([]);
  const [activeAgents, setActiveAgents] = useState<string[]>([]);
  const [currentLatency, setCurrentLatency] = useState<number>(0);
  const [lastAnnotatedBase64, setLastAnnotatedBase64] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<any>(null);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsStreaming(true);
      setIsPaused(false);
    } catch (err: any) {
      console.error(err);
      if (err.name === 'NotAllowedError') {
        setError("Camera permission denied. Please allow camera access in browser settings.");
      } else if (err.name === 'NotFoundError') {
        setError("No camera device detected on this system.");
      } else {
        setError(`Camera access error: ${err.message}`);
      }
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  // Inference capture loop
  useEffect(() => {
    if (isStreaming && !isPaused) {
      intervalRef.current = setInterval(async () => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video.videoWidth === 0 || video.videoHeight === 0) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const base64Data = canvas.toDataURL('image/jpeg', 0.85);

        try {
          const res = await api.processCameraFrame(base64Data, query, Date.now() / 1000);
          if (res.status === 'success') {
            setLiveFindings(res.findings || []);
            setActiveAgents(res.executed_agents || []);
            setCurrentLatency(res.latency_ms || 0);
            if (res.annotated_frame_base64) {
              setLastAnnotatedBase64(res.annotated_frame_base64);
            }
          }
        } catch (e) {
          console.error("Camera frame inference failed:", e);
        }
      }, 1500); // Sample every 1.5s
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isStreaming, isPaused, query]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-6 py-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Camera className="w-5 h-5 text-accent-emerald" />
            <span>Real-Time Webcam Vision Intelligence</span>
          </h2>
          <p className="text-xs text-graphite-400 font-mono">
            Low-latency frame sampling sending video payloads directly to the backend multi-agent pipeline.
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-3">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="flex items-center space-x-2 bg-accent-emerald hover:bg-emerald-600 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-md transition-all font-mono"
            >
              <Play className="w-4 h-4" />
              <span>Start Camera</span>
            </button>
          ) : (
            <>
              <button
                onClick={togglePause}
                className="flex items-center space-x-1.5 bg-graphite-700 hover:bg-graphite-600 text-white text-xs font-mono px-3 py-2 rounded-lg transition-all"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                <span>{isPaused ? "Resume" : "Pause"}</span>
              </button>
              <button
                onClick={stopCamera}
                className="flex items-center space-x-1.5 bg-red-600/80 hover:bg-red-600 text-white text-xs font-mono px-3 py-2 rounded-lg transition-all"
              >
                <Square className="w-4 h-4" />
                <span>Stop</span>
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-xs text-red-300 flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Query Bar */}
      <div className="tech-panel rounded-xl p-4 mb-6 flex items-center space-x-4">
        <span className="text-xs font-mono text-graphite-400 whitespace-nowrap">Active Inspection Goal:</span>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 px-3 py-1.5 rounded-lg border border-graphite-700 bg-graphite-800 text-xs text-white"
        />
        <div className="flex items-center space-x-2 text-xs font-mono text-graphite-400 bg-graphite-800/80 px-3 py-1.5 rounded border border-graphite-700">
          <Clock className="w-3.5 h-3.5 text-accent-cyan" />
          <span>{currentLatency.toFixed(1)} ms</span>
        </div>
      </div>

      {/* Camera Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Video Viewports */}
        <div className="lg:col-span-8 space-y-4">
          <div className="tech-panel rounded-xl p-4 overflow-hidden relative">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="text-[11px] font-mono text-graphite-400 block mb-2">Live Camera Feed</span>
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  className="w-full h-64 object-cover rounded-lg border border-graphite-800 bg-black"
                />
              </div>
              <div>
                <span className="text-[11px] font-mono text-accent-cyan block mb-2">Annotated Inference Frame</span>
                {lastAnnotatedBase64 ? (
                  <img
                    src={lastAnnotatedBase64}
                    alt="Annotated Live Frame"
                    className="w-full h-64 object-cover rounded-lg border border-accent-blue/30 shadow-md"
                  />
                ) : (
                  <div className="w-full h-64 flex items-center justify-center text-xs text-graphite-500 font-mono border border-dashed border-graphite-800 rounded-lg">
                    Awaiting live inference frame...
                  </div>
                )}
              </div>
            </div>

            {/* Hidden canvas for frame extraction */}
            <canvas ref={canvasRef} className="hidden" />
          </div>

          {/* Active Agents Running Badge Bar */}
          <div className="tech-panel rounded-xl p-4 flex items-center space-x-3 overflow-x-auto text-xs font-mono">
            <span className="text-graphite-400 whitespace-nowrap">Active Specialists:</span>
            {activeAgents.length > 0 ? (
              activeAgents.map((ag) => (
                <span key={ag} className="px-2.5 py-1 rounded bg-accent-blue/20 text-accent-cyan border border-accent-blue/30 font-semibold uppercase">
                  {ag}
                </span>
              ))
            ) : (
              <span className="text-graphite-500">Idle</span>
            )}
          </div>
        </div>

        {/* Right: Live Findings List */}
        <div className="lg:col-span-4">
          <div className="tech-panel rounded-xl p-5 h-full">
            <h3 className="text-xs font-mono uppercase tracking-wider text-white font-bold mb-3 flex items-center justify-between">
              <span>Live Detected Entities ({liveFindings.length})</span>
              {isStreaming && <span className="w-2 h-2 rounded-full bg-accent-emerald animate-ping" />}
            </h3>

            {liveFindings.length === 0 ? (
              <p className="text-xs text-graphite-500 font-mono text-center py-12">
                Start camera to observe real-time findings.
              </p>
            ) : (
              <div className="space-y-2.5 max-h-[460px] overflow-y-auto">
                {liveFindings.map((f, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-graphite-800/60 border border-graphite-700/80 text-xs">
                    <div className="flex items-center justify-between mb-1 font-mono">
                      <span className="font-bold text-white">{f.label}</span>
                      {f.detection_confidence && (
                        <span className="text-accent-cyan">{(f.detection_confidence * 100).toFixed(0)}%</span>
                      )}
                    </div>
                    <p className="text-[11px] text-graphite-300 font-sans">{f.observed_evidence}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
