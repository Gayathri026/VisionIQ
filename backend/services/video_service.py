import os
import cv2
import time
import uuid
from typing import Dict, Any, List, Optional
from backend.config import settings
from backend.orchestrator.graph import vision_iq_app

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

class VideoTracker:
    """Simple and robust IoU-based multi-object tracker to avoid double counting across frames."""
    def __init__(self, iou_threshold: float = 0.35):
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]], frame_time: float) -> List[Dict[str, Any]]:
        updated_detections = []
        unmatched_detections = list(range(len(detections)))
        matched_tracks = set()

        # Match existing active tracks to new detections
        for track_id, track_info in self.active_tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                if det["label"] == track_info["label"]:
                    iou = compute_box_iou(det["bbox"], track_info["bbox"])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_det_idx = det_idx

            if best_det_idx != -1:
                det = detections[best_det_idx]
                det["track_id"] = track_id
                self.active_tracks[track_id] = {
                    "bbox": det["bbox"],
                    "label": det["label"],
                    "last_seen": frame_time
                }
                matched_tracks.add(track_id)
                unmatched_detections.remove(best_det_idx)
                updated_detections.append(det)

        # Assign new track IDs for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            track_id = self.next_track_id
            self.next_track_id += 1
            det["track_id"] = track_id
            self.active_tracks[track_id] = {
                "bbox": det["bbox"],
                "label": det["label"],
                "last_seen": frame_time
            }
            updated_detections.append(det)

        return updated_detections

class VideoProcessingService:
    """
    Streaming video frame processor:
    - Samples frames at configurable intervals.
    - Runs VisionIQ orchestration on sampled frames.
    - Tracks objects to prevent duplicate counting.
    - Emits progress and aggregated findings.
    """
    def __init__(self):
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        os.makedirs(settings.ANNOTATIONS_DIR, exist_ok=True)

    def process_video(
        self,
        video_path: str,
        query: str,
        sample_fps: float = 1.0,
        max_frames: int = 30
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_video_frames / video_fps if video_fps > 0 else 0.0

        # Frame step interval
        frame_interval = max(1, int(video_fps / sample_fps))
        
        tracker = VideoTracker(iou_threshold=0.35)
        
        sampled_frame_results = []
        unique_entities_tracked = set()
        aggregated_findings = []
        
        current_frame_idx = 0
        processed_count = 0

        # Process frames sequentially without loading all into memory
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame_idx % frame_interval == 0:
                timestamp_sec = round(current_frame_idx / video_fps, 2)
                
                # Save temp frame for model inference
                frame_filename = f"temp_frame_{uuid.uuid4().hex[:8]}.jpg"
                temp_frame_path = os.path.join(settings.UPLOADS_DIR, frame_filename)
                cv2.imwrite(temp_frame_path, frame)

                try:
                    state = {
                        "image_path": temp_frame_path,
                        "user_query": query,
                        "input_type": "video"
                    }
                    res_state = vision_iq_app.invoke(state)
                    resp = res_state.get("final_response", {})
                    
                    frame_findings = resp.get("findings", [])
                    # Assign tracking IDs to findings
                    tracked_findings = tracker.update(frame_findings, timestamp_sec)

                    for tf in tracked_findings:
                        tf["frame_timestamp"] = timestamp_sec
                        if tf.get("track_id"):
                            unique_entities_tracked.add((tf["label"], tf["track_id"]))
                    
                    sampled_frame_results.append({
                        "frame_index": current_frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "annotated_path": resp.get("annotated_path"),
                        "findings_count": len(tracked_findings),
                        "findings": tracked_findings
                    })
                    aggregated_findings.extend(tracked_findings)
                    processed_count += 1
                finally:
                    # Clean up temporary frame file to prevent disk exhaustion
                    if os.path.exists(temp_frame_path):
                        try:
                            os.remove(temp_frame_path)
                        except Exception:
                            pass

            current_frame_idx += 1

        cap.release()

        # Deduplicate and summarize unique tracked findings
        entity_summary = {}
        for label, track_id in unique_entities_tracked:
            entity_summary[label] = entity_summary.get(label, 0) + 1

        total_latency = (time.time() - start_time) * 1000.0

        return {
            "video_path": video_path,
            "duration_sec": round(duration_sec, 2),
            "total_video_frames": total_video_frames,
            "sampled_frames_processed": len(sampled_frame_results),
            "sample_fps": sample_fps,
            "unique_tracked_entities": entity_summary,
            "total_unique_entities": len(unique_entities_tracked),
            "frame_results": sampled_frame_results,
            "aggregated_findings": aggregated_findings,
            "total_latency_ms": round(total_latency, 2)
        }

video_service = VideoProcessingService()
