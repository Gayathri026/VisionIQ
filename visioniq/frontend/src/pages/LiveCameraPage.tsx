import React, { useState, useRef, useEffect } from 'react';
import { api } from '../services/api';
import { Camera, Play, Pause, Square, AlertCircle, Cpu, Clock, RefreshCw, CheckCircle2, ShieldCheck, Activity, Video } from 'lucide-react';

export const LiveCameraPage: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [query, setQuery] = useState("Continuous PPE safety compliance monitoring");
  const [sampleIntervalMs, setSampleIntervalMs] = useState(1500);
  const [error, setError] = useState<string | null>(null);
  const [permissionState, setPermissionState] = useState<'idle' | 'requesting' | 'granted' | 'denied'>('idle');

  const [liveFindings, setLiveFindings] = useState<any[]>([]);
  const [activeAgents, setActiveAgents] = useState<string[]>([]);
  const [detectedIntent, setDetectedIntent] = useState<string | null>(null);
  const [currentLatency, setCurrentLatency] = useState<number>(0);
  const [lastAnnotatedBase64, setLastAnnotatedBase64] = useState<string | null>(null);
  const [fpsCounter, setFpsCounter] = useState<number>(0);
  const [totalFramesProcessed, setTotalFramesProcessed] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<any>(null);
  const isInferringRef = useRef<boolean>(false);

  const startCamera = async () => {
    setError(null);
    setPermissionState('requesting');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("WebRTC MediaDevices API is not supported in this browser environment.");
      setPermissionState('denied');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setPermissionState('granted');
      setIsStreaming(true);
      setIsPaused(false);
    } catch (err: any) {
      console.error("Camera access error:", err);
      setPermissionState('denied');
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError("Camera permission denied. Please grant camera access in browser site settings.");
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError("No camera hardware detected on this machine.");
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError("Camera device is currently locked or in use by another application.");
      } else {
        setError(`Camera error: ${err.message || 'Unable to access media stream'}`);
      }
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
    setPermissionState('idle');
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  // Inference capture loop with busy lock
  useEffect(() => {
    if (isStreaming && !isPaused) {
      intervalRef.current = setInterval(async () => {
        if (isInferringRef.current) return;
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

        isInferringRef.current = true;
        const reqTimestamp = Date.now() / 1000;

        try {
          const res = await api.processCameraFrame(base64Data, query, reqTimestamp);
          if (res.status === 'success') {
            setLiveFindings(res.findings || []);
            setActiveAgents(res.executed_agents || []);
            setDetectedIntent(res.detected_intent || null);
            setCurrentLatency(res.latency_ms || 0);
            if (res.annotated_frame_base64) {
              setLastAnnotatedBase64(res.annotated_frame_base64);
            }
            setTotalFramesProcessed((prev) => prev + 1);
            setFpsCounter((prev) => (res.latency_ms > 0 ? +(1000.0 / res.latency_ms).toFixed(1) : 0));
          }
        } catch (e: any) {
          console.error("Camera frame inference failed:", e);
        } finally {
          isInferringRef.current = false;
        }
      }, sampleIntervalMs);
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
  }, [isStreaming, isPaused, query, sampleIntervalMs]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="max-w-[1760px] mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="command-panel p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-[#34D399]/15 border border-[#34D399]/30 flex items-center justify-center text-[#34D399]">
            <Camera className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold text-[#F1F5F9] tracking-tight">
              Real-Time Webcam Multi-Agent Vision Intelligence
            </h1>
            <p className="text-xs font-mono text-[#8998AF]">
              Direct WebRTC Frame Capture • Dynamic Agent Selection • Live Bounding Box Projection
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="flex items-center space-x-2 bg-gradient-to-r from-[#059669] to-[#34D399] hover:opacity-90 text-[#0B1020] text-xs font-extrabold px-5 py-2.5 rounded-lg shadow-md transition-all font-mono"
            >
              <Play className="w-4 h-4 text-[#0B1020]" />
              <span>Start Live Stream</span>
            </button>
          ) : (
            <>
              <button
                onClick={togglePause}
                className="flex items-center space-x-1.5 bg-[#1B2942] hover:bg-[#223252] border border-[#2B3A54] text-white text-xs font-mono px-3.5 py-2.5 rounded-lg transition-all"
              >
                {isPaused ? <Play className="w-4 h-4 text-[#34D399]" /> : <Pause className="w-4 h-4 text-[#FBBF24]" />}
                <span>{isPaused ? "Resume Inference" : "Pause Stream"}</span>
              </button>
              <button
                onClick={stopCamera}
                className="flex items-center space-x-1.5 bg-[#FB7185]/20 hover:bg-[#FB7185]/30 border border-[#FB7185]/50 text-[#FB7185] text-xs font-mono px-3.5 py-2.5 rounded-lg transition-all"
              >
                <Square className="w-4 h-4" />
                <span>Stop Stream & Release</span>
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-[#FB7185]/15 border border-[#FB7185]/40 text-xs text-[#FB7185] flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-[#FB7185] flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Query Bar and Sampling Interval */}
      <div className="command-panel p-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          <div className="md:col-span-6 flex items-center space-x-2">
            <span className="text-xs font-mono text-[#8998AF] whitespace-nowrap">Goal / Query:</span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 px-3.5 py-2 rounded-lg border border-[#2B3A54] bg-[#10182B] text-xs text-white focus:outline-none focus:border-[#43D9F5]"
            />
          </div>

          <div className="md:col-span-3">
            <label className="block text-[11px] font-mono text-[#8998AF] mb-1">
              Sample Interval: <span className="text-[#43D9F5] font-bold">{sampleIntervalMs} ms</span>
            </label>
            <input
              type="range"
              min="500"
              max="3000"
              step="250"
              value={sampleIntervalMs}
              onChange={(e) => setSampleIntervalMs(parseInt(e.target.value))}
              className="w-full accent-[#43D9F5] cursor-pointer h-1.5 bg-[#2B3A54] rounded-lg"
            />
          </div>

          <div className="md:col-span-3 flex items-center justify-end space-x-3 text-xs font-mono">
            <div className="flex items-center space-x-1.5 bg-[#10182B] px-3 py-2 rounded-lg border border-[#2B3A54] text-[#B4C0D4]">
              <Clock className="w-3.5 h-3.5 text-[#43D9F5]" />
              <span>{currentLatency.toFixed(1)} ms</span>
            </div>
            <div className="flex items-center space-x-1.5 bg-[#10182B] px-3 py-2 rounded-lg border border-[#2B3A54] text-[#34D399]">
              <Activity className="w-3.5 h-3.5" />
              <span>{totalFramesProcessed} frames</span>
            </div>
          </div>
        </div>
      </div>

      {/* Camera Grid Layout (Video Sized to Content, Independent of Right Side) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Video Viewports Sized to Stream (self-start + sticky) */}
        <div className="lg:col-span-8 space-y-4 self-start lg:sticky lg:top-20">
          <div className="command-panel p-4 overflow-hidden relative">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-[#8998AF] font-semibold">Raw Camera Stream</span>
                  {isStreaming && (
                    <span className="flex items-center space-x-1 text-[11px] font-mono text-[#34D399] font-bold">
                      <span className="w-2 h-2 rounded-full bg-[#34D399] animate-pulse" />
                      <span>LIVE</span>
                    </span>
                  )}
                </div>
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  className="w-full h-80 object-cover rounded-lg border border-[#2B3A54] bg-[#070A14]"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-[#43D9F5] font-semibold">Annotated Inference Viewport</span>
                  {detectedIntent && (
                    <span className="text-[11px] font-mono text-[#8998AF] uppercase font-bold">
                      Intent: {detectedIntent}
                    </span>
                  )}
                </div>
                {lastAnnotatedBase64 ? (
                  <img
                    src={lastAnnotatedBase64}
                    alt="Annotated Live Frame"
                    className="w-full h-80 object-cover rounded-lg border border-[#43D9F5]/40 shadow-xl shadow-[#43D9F5]/10"
                  />
                ) : (
                  <div className="w-full h-80 flex items-center justify-center text-xs text-[#8998AF] font-mono border border-dashed border-[#2B3A54] rounded-lg bg-[#070A14]">
                    {isStreaming ? "Capturing & inferring frames..." : "Start camera to stream live inference"}
                  </div>
                )}
              </div>
            </div>

            {/* Hidden canvas for frame extraction */}
            <canvas ref={canvasRef} className="hidden" />
          </div>

          {/* Active Agents Running Badge Bar */}
          <div className="command-panel p-4 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center space-x-2">
              <span className="text-[#8998AF] whitespace-nowrap">Active Vision Specialists:</span>
              {activeAgents.length > 0 ? (
                activeAgents.map((ag) => (
                  <span key={ag} className="px-2.5 py-1 rounded bg-[#10182B] text-[#43D9F5] border border-[#43D9F5]/40 font-bold uppercase">
                    {ag}
                  </span>
                ))
              ) : (
                <span className="text-[#8998AF]">Awaiting stream activation</span>
              )}
            </div>

            <div className="text-[#8998AF] text-xs">
              Status: <span className="text-white font-bold capitalize">{permissionState}</span>
            </div>
          </div>
        </div>

        {/* Right: Live Findings List */}
        <div className="lg:col-span-4">
          <div className="command-panel p-5 h-full space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-wider text-white font-bold pb-2 border-b border-[#2B3A54] flex items-center justify-between">
              <span>Live Detected Entities ({liveFindings.length})</span>
              {isStreaming && !isPaused && <span className="w-2.5 h-2.5 rounded-full bg-[#34D399] animate-ping" />}
            </h3>

            {liveFindings.length === 0 ? (
              <p className="text-xs text-[#8998AF] font-mono text-center py-20">
                {isStreaming ? "No entities detected in current frame." : "Start camera to observe real-time findings."}
              </p>
            ) : (
              <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
                {liveFindings.map((f, idx) => (
                  <div key={idx} className="p-3.5 rounded-lg bg-[#10182B] border border-[#2B3A54] text-xs space-y-1">
                    <div className="flex items-center justify-between font-mono">
                      <span className="font-bold text-white text-sm">{f.label}</span>
                      {f.detection_confidence && (
                        <span className="text-[#43D9F5] font-bold">{(f.detection_confidence * 100).toFixed(0)}%</span>
                      )}
                    </div>
                    <p className="text-xs text-[#B4C0D4] font-sans leading-relaxed">{f.observed_evidence}</p>
                    {f.model_name && (
                      <div className="text-[11px] font-mono text-[#8998AF]">
                        {f.model_name}
                      </div>
                    )}
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
