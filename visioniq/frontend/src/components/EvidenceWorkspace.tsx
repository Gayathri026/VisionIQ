import React, { useState, useRef, useEffect } from 'react';
import { Finding } from '../types';
import {
  Eye,
  Layers,
  FileText,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Info,
  SlidersHorizontal,
  ChevronRight,
  Crosshair
} from 'lucide-react';

interface Props {
  originalUrl?: string;
  annotatedUrl?: string;
  pdfUrl?: string;
  findings: Finding[];
  selectedFindingUid?: string | null;
  onSelectFinding: (uid: string | null) => void;
  imageDimensions?: { width: number; height: number };
}

export const EvidenceWorkspace: React.FC<Props> = ({
  originalUrl,
  annotatedUrl,
  pdfUrl,
  findings,
  selectedFindingUid,
  onSelectFinding,
  imageDimensions
}) => {
  // Viewer state
  const [viewMode, setViewMode] = useState<'annotated' | 'original' | 'split'>('annotated');
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [minConfidenceFilter, setMinConfidenceFilter] = useState<number>(0.20);
  const [splitPosition, setSplitPosition] = useState<number>(50); // percentage for split view

  // DOM Refs for coordinate transformation
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgRenderedSize, setImgRenderedSize] = useState<{ width: number; height: number; naturalWidth: number; naturalHeight: number }>({
    width: 0,
    height: 0,
    naturalWidth: 1,
    naturalHeight: 1
  });

  const selectedFinding = findings.find(f => f.finding_uid === selectedFindingUid);

  // Update rendered image dimensions on load and resize
  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImgRenderedSize({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth || 1,
      naturalHeight: img.naturalHeight || 1
    });
  };

  useEffect(() => {
    const updateSize = () => {
      if (imgRef.current) {
        setImgRenderedSize({
          width: imgRef.current.clientWidth,
          height: imgRef.current.clientHeight,
          naturalWidth: imgRef.current.naturalWidth || 1,
          naturalHeight: imgRef.current.naturalHeight || 1
        });
      }
    };
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Zoom controls
  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.25, 4.0));
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => {
    setZoomLevel(1.0);
    setPanOffset({ x: 0, y: 0 });
  };

  // Pan interaction
  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoomLevel > 1.0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoomLevel > 1.0) {
      setPanOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Toggle Fullscreen
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen?.().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen?.().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  // Empty state
  if (!originalUrl && !annotatedUrl) {
    return (
      <div className="command-panel rounded-xl p-10 text-center flex flex-col items-center justify-center border-dashed border-[#2B3A54] min-h-[340px]">
        <div className="w-16 h-16 rounded-2xl bg-[#1B2942] border border-[#2B3A54] flex items-center justify-center text-[#43D9F5] mb-4 shadow-lg shadow-black/40">
          <Layers className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-[#F1F5F9]">No Inspection Media Loaded</h3>
        <p className="text-base text-[#8998AF] max-w-md mt-2 leading-relaxed">
          Upload an industrial factory photograph, PPE scene, or metal plate to activate real-time multi-agent perception and evidence synthesis.
        </p>
      </div>
    );
  }

  // Filter findings based on confidence cutoff
  const visibleFindings = findings.filter(f => {
    const conf = f.detection_confidence ?? (1 - (f.anomaly_score ?? 0));
    return conf >= minConfidenceFilter;
  });

  return (
    <div
      ref={containerRef}
      className={`command-panel rounded-xl overflow-hidden flex flex-col transition-all ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none bg-[#0B1020]' : 'w-full'
      }`}
    >
      {/* Top Controls Header Bar */}
      <div className="px-5 py-3 border-b border-[#2B3A54] bg-[#10182B] flex flex-wrap items-center justify-between gap-3">
        {/* Left: View Mode Toggles */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs font-mono text-[#43D9F5] font-bold uppercase tracking-wider">
            <Eye className="w-4 h-4 text-[#43D9F5]" />
            <span>Inspection Canvas</span>
          </div>

          <div className="bg-[#151F35] p-1 rounded-lg border border-[#2B3A54] flex text-xs font-medium">
            <button
              onClick={() => setViewMode('annotated')}
              className={`px-3 py-1.5 rounded-md transition-all ${
                viewMode === 'annotated'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 font-semibold shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Annotated
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-3 py-1.5 rounded-md transition-all ${
                viewMode === 'original'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 font-semibold shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1.5 rounded-md transition-all ${
                viewMode === 'split'
                  ? 'bg-[#1B2942] text-white border border-[#43D9F5]/40 font-semibold shadow-sm'
                  : 'text-[#8998AF] hover:text-white'
              }`}
            >
              Split View
            </button>
          </div>
        </div>

        {/* Center/Right: Zoom, Pan, Overlay & Fullscreen Controls */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          {/* Zoom controls */}
          <div className="flex items-center bg-[#151F35] border border-[#2B3A54] rounded-lg p-0.5 text-[#B4C0D4]">
            <button
              onClick={handleZoomOut}
              className="p-1.5 hover:text-white hover:bg-[#1B2942] rounded transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="px-2 text-[11px] font-bold text-[#F1F5F9]">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-1.5 hover:text-white hover:bg-[#1B2942] rounded transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 hover:text-white hover:bg-[#1B2942] rounded transition-colors border-l border-[#2B3A54]"
              title="Reset Zoom & Pan"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Overlays Toggle */}
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            className={`px-2.5 py-1.5 rounded-lg border text-[12px] font-medium transition-all ${
              showOverlays
                ? 'bg-[#1B2942] text-[#43D9F5] border-[#43D9F5]/40'
                : 'bg-[#151F35] text-[#8998AF] border-[#2B3A54] hover:text-white'
            }`}
            title="Toggle Bounding Boxes"
          >
            Boxes: {showOverlays ? 'ON' : 'OFF'}
          </button>

          {/* Fullscreen Button */}
          <button
            onClick={toggleFullscreen}
            className="p-1.5 rounded-lg bg-[#151F35] hover:bg-[#1B2942] border border-[#2B3A54] text-[#B4C0D4] hover:text-white transition-colors"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen View"}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>

          {/* Export PDF Button */}
          {pdfUrl && (
            <a
              href={pdfUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#43D9F5]/20 hover:bg-[#43D9F5]/30 border border-[#43D9F5]/60 text-[#43D9F5] font-semibold text-xs transition-colors shadow-sm"
              title="Download Formal PDF Report"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>PDF Report</span>
            </a>
          )}
        </div>
      </div>

      {/* Interactive Active Entity Focus Pill Banner */}
      {selectedFinding && (
        <div className="px-5 py-2.5 bg-[#1B2942] border-b border-[#3A4D6F] flex items-center justify-between text-xs font-mono shadow-sm">
          <div className="flex items-center space-x-2.5 truncate">
            <span className="w-2.5 h-2.5 rounded-full bg-[#43D9F5] animate-ping flex-shrink-0"></span>
            <span className="text-[#43D9F5] font-bold text-sm">{selectedFinding.finding_uid}:</span>
            <span className="text-[#F1F5F9] font-semibold text-sm truncate">{selectedFinding.label}</span>
            {selectedFinding.detection_confidence != null && (
              <span className="bg-[#151F35] text-[#34D399] border border-[#34D399]/30 px-2 py-0.5 rounded text-[11px] font-bold">
                {(selectedFinding.detection_confidence * 100).toFixed(1)}% Conf
              </span>
            )}
            {selectedFinding.spatial_zone && (
              <span className="bg-[#151F35] text-[#818CF8] border border-[#818CF8]/30 px-2 py-0.5 rounded text-[11px]">
                Zone: {selectedFinding.spatial_zone}
              </span>
            )}
          </div>
          <button
            onClick={() => onSelectFinding(null)}
            className="text-xs text-[#8998AF] hover:text-white bg-[#151F35] border border-[#2B3A54] px-2 py-1 rounded transition-colors flex-shrink-0"
          >
            Clear Focus
          </button>
        </div>
      )}

      {/* Main Inspection Canvas Surface (Adjusted dynamically to image aspect ratio) */}
      <div
        className="relative bg-[#070A14] flex items-center justify-center overflow-hidden p-3 select-none rounded-b-xl"
        style={{ cursor: zoomLevel > 1.0 ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        {viewMode === 'split' ? (
          /* Split Comparison View: Side-by-Side */
          <div className="grid grid-cols-2 gap-3 w-full items-center justify-center p-2">
            <div className="flex flex-col items-center justify-center bg-[#0B1020]/60 rounded-xl p-2 border border-[#2B3A54]">
              <span className="text-xs font-mono text-[#8998AF] mb-1.5 uppercase tracking-wider font-semibold">
                Original Input Raster
              </span>
              <img
                src={originalUrl}
                alt="Original Factory Capture"
                className="max-h-[min(55vh,480px)] max-w-full w-auto h-auto object-contain rounded-lg border border-[#2B3A54] shadow-xl block"
              />
            </div>
            <div className="flex flex-col items-center justify-center bg-[#0B1020]/60 rounded-xl p-2 border border-[#43D9F5]/30">
              <span className="text-xs font-mono text-[#43D9F5] mb-1.5 uppercase tracking-wider font-semibold flex items-center space-x-1.5">
                <Crosshair className="w-3.5 h-3.5" />
                <span>Annotated Multi-Agent Evidence</span>
              </span>
              <img
                src={annotatedUrl || originalUrl}
                alt="Annotated Model Detections"
                className="max-h-[min(55vh,480px)] max-w-full w-auto h-auto object-contain rounded-lg border border-[#43D9F5]/40 shadow-xl shadow-[#43D9F5]/5 block"
              />
            </div>
          </div>
        ) : (
          /* Single Canvas View: Sized strictly based on the image */
          <div
            className="relative inline-flex items-center justify-center transition-transform duration-75 max-w-full"
            style={{
              transform: `scale(${zoomLevel}) translate(${panOffset.x / zoomLevel}px, ${panOffset.y / zoomLevel}px)`
            }}
          >
            <img
              ref={imgRef}
              src={viewMode === 'annotated' ? (annotatedUrl || originalUrl) : originalUrl}
              alt="Industrial Inspection Evidence"
              onLoad={handleImageLoad}
              className="max-h-[min(65vh,560px)] max-w-full w-auto h-auto object-contain rounded-lg border border-[#2B3A54] shadow-2xl block"
            />

            {/* Interactive SVG Bounding Box Layer (Synchronized with Findings) */}
            {showOverlays && viewMode === 'annotated' && imgRenderedSize.width > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-auto"
                style={{ width: imgRenderedSize.width, height: imgRenderedSize.height }}
                viewBox={`0 0 ${imgRenderedSize.width} ${imgRenderedSize.height}`}
              >
                {visibleFindings.map((finding) => {
                  if (!finding.bbox || finding.bbox.length !== 4) return null;

                  // Normalize coordinates if needed
                  const [b1, b2, b3, b4] = finding.bbox;
                  let x1: number, y1: number, x2: number, y2: number;

                  if (b3 <= 1.0 && b4 <= 1.0) {
                    // Normalized [0, 1] coordinates
                    x1 = b1 * imgRenderedSize.width;
                    y1 = b2 * imgRenderedSize.height;
                    x2 = b3 * imgRenderedSize.width;
                    y2 = b4 * imgRenderedSize.height;
                  } else {
                    // Pixel coordinates based on natural image size
                    const scaleX = imgRenderedSize.width / imgRenderedSize.naturalWidth;
                    const scaleY = imgRenderedSize.height / imgRenderedSize.naturalHeight;
                    x1 = b1 * scaleX;
                    y1 = b2 * scaleY;
                    x2 = b3 * scaleX;
                    y2 = b4 * scaleY;
                  }

                  const boxW = Math.max(x2 - x1, 10);
                  const boxH = Math.max(y2 - y1, 10);
                  const isSelected = selectedFindingUid === finding.finding_uid;

                  // Determine box color
                  const isCritical = finding.label.toLowerCase().includes('violation') ||
                                     finding.label.toLowerCase().includes('no-') ||
                                     finding.is_confirmed_defect === true;
                  const isWarning = finding.human_review_required || finding.uncertainty_flag;

                  let strokeColor = '#43D9F5';
                  let fillColor = 'rgba(67, 217, 245, 0.12)';

                  if (isCritical) {
                    strokeColor = '#FB7185';
                    fillColor = 'rgba(251, 113, 133, 0.18)';
                  } else if (isWarning) {
                    strokeColor = '#FBBF24';
                    fillColor = 'rgba(251, 191, 36, 0.15)';
                  } else if (finding.label.toLowerCase().includes('compliant')) {
                    strokeColor = '#34D399';
                    fillColor = 'rgba(52, 211, 153, 0.12)';
                  }

                  if (isSelected) {
                    strokeColor = '#43D9F5';
                    fillColor = 'rgba(67, 217, 245, 0.35)';
                  }

                  return (
                    <g
                      key={finding.finding_uid}
                      onClick={() => onSelectFinding(isSelected ? null : finding.finding_uid)}
                      className="cursor-pointer transition-all duration-150 group"
                    >
                      {/* Bounding Box Rect */}
                      <rect
                        x={x1}
                        y={y1}
                        width={boxW}
                        height={boxH}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={isSelected ? 3.5 : 2}
                        strokeDasharray={isSelected ? '6,3' : 'none'}
                        className="transition-all hover:stroke-white"
                      />

                      {/* Technical Corner Accents */}
                      {isSelected && (
                        <>
                          <line x1={x1 - 4} y1={y1} x2={x1 + 8} y2={y1} stroke="#43D9F5" strokeWidth={3} />
                          <line x1={x1} y1={y1 - 4} x2={x1} y2={y1 + 8} stroke="#43D9F5" strokeWidth={3} />
                          <line x1={x2 + 4} y1={y2} x2={x2 - 8} y2={y2} stroke="#43D9F5" strokeWidth={3} />
                          <line x1={x2} y1={y2 + 4} x2={x2} y2={y2 - 8} stroke="#43D9F5" strokeWidth={3} />
                        </>
                      )}

                      {/* Floating Box Tag Badge */}
                      {showLabels && (
                        <g transform={`translate(${x1}, ${Math.max(y1 - 22, 4)})`}>
                          <rect
                            x={0}
                            y={0}
                            width={Math.min(boxW, 140)}
                            height={20}
                            rx={4}
                            fill="#10182B"
                            stroke={strokeColor}
                            strokeWidth={1}
                          />
                          <text
                            x={6}
                            y={14}
                            fill="#F1F5F9"
                            fontSize="11px"
                            fontWeight="bold"
                            fontFamily="JetBrains Mono, monospace"
                          >
                            {finding.finding_uid}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        )}
      </div>

      {/* Bottom Entity Quick-Selection Scroller */}
      {findings.length > 0 && (
        <div className="px-5 py-3 border-t border-[#2B3A54] bg-[#10182B] flex items-center space-x-2.5 overflow-x-auto">
          <span className="text-xs font-mono text-[#8998AF] uppercase tracking-wider whitespace-nowrap font-semibold">
            Interactive Entities ({visibleFindings.length}):
          </span>
          {visibleFindings.map((f) => {
            const isSelected = selectedFindingUid === f.finding_uid;
            return (
              <button
                key={f.finding_uid}
                onClick={() => onSelectFinding(isSelected ? null : f.finding_uid)}
                className={`text-xs font-mono px-3 py-1.5 rounded-lg transition-all whitespace-nowrap border flex items-center space-x-1.5 ${
                  isSelected
                    ? 'bg-[#1B2942] text-white border-[#43D9F5] shadow-sm shadow-[#43D9F5]/20 font-bold'
                    : 'bg-[#151F35] text-[#B4C0D4] border-[#2B3A54] hover:border-[#3A4D6F] hover:text-white'
                }`}
              >
                <span className="text-[#43D9F5] font-bold">{f.finding_uid}</span>
                <span className="text-[#8998AF]">•</span>
                <span className="truncate max-w-[140px]">{f.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
