import os
import cv2
import time
import math
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.orchestration.router import route_query
from backend.app.agents.agent_b_object import ObjectDetectionAgent
from backend.app.agents.agent_c_ppe import PPESafetyAgent
from backend.app.agents.agent_e_defect import IndustrialDefectAgent
from backend.app.services.annotation_service import draw_styled_box

object_agent = ObjectDetectionAgent()
ppe_agent = PPESafetyAgent()
defect_agent = IndustrialDefectAgent()

class ActiveTrack:
    """Represents a spatio-temporal entity tracked across video frames."""
    def __init__(self, track_id: int, class_label: str, bbox: List[float], confidence: float, frame_idx: int, timestamp: float):
        self.track_id: int = track_id
        self.class_label: str = class_label
        self.current_bbox: List[float] = bbox
        self.confidence: float = confidence
        self.start_frame: int = frame_idx
        self.last_frame: int = frame_idx
        self.start_time: float = timestamp
        self.last_time: float = timestamp
        self.frames_seen: int = 1
        self.missing_streak: int = 0
        
        # Safety & compliance temporal tracking
        self.ppe_violation_count: int = 0
        self.observed_states: List[Dict[str, Any]] = []

    def update(self, bbox: List[float], confidence: float, frame_idx: int, timestamp: float, ppe_violation: bool = False, state_info: Optional[Dict[str, Any]] = None):
        self.current_bbox = bbox
        self.confidence = confidence
        self.last_frame = frame_idx
        self.last_time = timestamp
        self.frames_seen += 1
        self.missing_streak = 0
        if ppe_violation:
            self.ppe_violation_count += 1
        if state_info:
            self.observed_states.append(state_info)

    def mark_missed(self):
        self.missing_streak += 1

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def compute_centroid_distance_ratio(boxA: List[float], boxB: List[float]) -> float:
    """Normalized Euclidean distance between bounding box centroids relative to diagonal."""
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0

    diagA = math.sqrt((boxA[2] - boxA[0]) ** 2 + (boxA[3] - boxA[1]) ** 2) or 1.0
    dist = math.sqrt((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2)
    return dist / diagA

class VideoTrackingService:
    """
    High-performance spatio-temporal video tracking and incremental frame inference engine.
    Processes long videos incrementally to avoid memory exhaustion, provides multi-object
    tracking (IoU + Centroid matching), unique non-duplicate entity counts, and temporal
    violation persistence verification.
    """
    def __init__(self):
        self.max_lost_frames: int = 3
        self.iou_match_threshold: float = 0.25
        self.centroid_dist_threshold: float = 0.45

    def process_video(
        self,
        video_path: str,
        query: str = "Track personnel and verify safety compliance",
        sample_fps: float = 1.0,
        max_frames: int = 25
    ) -> Dict[str, Any]:
        """
        Incrementally processes video at sample_fps, executing query-routed agents on each sampled
        frame, tracking entities across time, and synthesizing unique counts and temporal violations.
        """
        t0 = time.time()
        session_id = str(uuid.uuid4())

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file. Verify codec compatibility (H.264/MP4).")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / video_fps, 2) if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains 0 playable frames.")

        # Determine frame indices to sample
        step_frames = max(1, int(round(video_fps / max(0.1, sample_fps))))
        sample_indices = []
        curr = 0
        while curr < total_frames and len(sample_indices) < max_frames:
            sample_indices.append(curr)
            curr += step_frames

        # Create output directory for annotated video frames
        video_ann_dir = os.path.join(settings.ANNOTATIONS_DIR, f"video_{session_id[:8]}")
        os.makedirs(video_ann_dir, exist_ok=True)

        # 1. Routing decision based on query
        routing_plan = route_query(query)
        selected_agents = routing_plan.get("selected_agents", ["object"])
        detected_intent = routing_plan.get("intent", "object_counting_detection")

        # 2. Tracking state
        active_tracks: List[ActiveTrack] = []
        all_tracks: List[ActiveTrack] = []
        next_track_id: int = 1

        frame_results: List[Dict[str, Any]] = []
        all_session_findings: List[Dict[str, Any]] = []

        try:
            for idx, frame_idx in enumerate(sample_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                ts_sec = round(frame_idx / video_fps, 2)

                # Save temporary frame raster for agent processing
                temp_frame_path = os.path.join(settings.UPLOADS_DIR, f"temp_frame_{session_id[:8]}_{idx}.jpg")
                cv2.imwrite(temp_frame_path, frame)

                frame_findings: List[Dict[str, Any]] = []
                frame_per_class_counts: Dict[str, int] = {}

                # Execute routed specialist agents on this frame
                if "object" in selected_agents:
                    obj_res = object_agent.analyze(image_path=temp_frame_path)
                    frame_findings.extend(obj_res.get("findings", []))

                if "ppe" in selected_agents:
                    ppe_res = ppe_agent.analyze(image_path=temp_frame_path, object_result={"findings": frame_findings})
                    frame_findings.extend(ppe_res.get("findings", []))

                if "defect" in selected_agents:
                    def_res = defect_agent.analyze(image_path=temp_frame_path)
                    frame_findings.extend(def_res.get("findings", []))

                # Count detections in this specific frame
                for f in frame_findings:
                    lbl = f.get("label", "unknown").lower()
                    clean_lbl = "person" if "person" in lbl or "worker" in lbl else lbl
                    frame_per_class_counts[clean_lbl] = frame_per_class_counts.get(clean_lbl, 0) + 1

                # Extract trackable objects (persons, machinery, vehicles)
                candidates: List[Dict[str, Any]] = []
                for f in frame_findings:
                    bbox = f.get("bbox")
                    if bbox and len(bbox) == 4:
                        lbl = f.get("label", "").lower()
                        if "person" in lbl or "worker" in lbl or f.get("agent_source") in ("object", "ppe"):
                            candidates.append({
                                "bbox": bbox,
                                "label": "person",
                                "conf": f.get("detection_confidence", 0.85),
                                "finding": f
                            })

                # Associate candidates with active tracks (IoU + Centroid)
                matched_candidate_indices = set()
                matched_track_indices = set()

                for t_idx, track in enumerate(active_tracks):
                    best_iou = 0.0
                    best_c_idx = -1
                    for c_idx, cand in enumerate(candidates):
                        if c_idx in matched_candidate_indices:
                            continue
                        iou = compute_iou(track.current_bbox, cand["bbox"])
                        c_dist = compute_centroid_distance_ratio(track.current_bbox, cand["bbox"])
                        
                        # Match condition: IoU >= threshold OR (IoU >= 0.15 AND close centroid)
                        if iou >= self.iou_match_threshold or (iou >= 0.12 and c_dist <= self.centroid_dist_threshold):
                            if iou > best_iou:
                                best_iou = iou
                                best_c_idx = c_idx

                    if best_c_idx >= 0:
                        cand = candidates[best_c_idx]
                        is_violation = ("NO-Hardhat" in cand["finding"].get("observed_evidence", "") or
                                        "NO-Safety Vest" in cand["finding"].get("observed_evidence", "") or
                                        "Violation" in cand["finding"].get("label", ""))
                        track.update(cand["bbox"], cand["conf"], frame_idx, ts_sec, ppe_violation=is_violation)
                        cand["finding"]["track_id"] = track.track_id
                        cand["finding"]["frame_timestamp"] = ts_sec
                        matched_candidate_indices.add(best_c_idx)
                        matched_track_indices.add(t_idx)

                # Update tracks that were not matched
                for t_idx, track in enumerate(active_tracks):
                    if t_idx not in matched_track_indices:
                        track.mark_missed()

                # Start new tracks for unmatched candidates
                for c_idx, cand in enumerate(candidates):
                    if c_idx not in matched_candidate_indices:
                        new_track = ActiveTrack(
                            track_id=next_track_id,
                            class_label=cand["label"],
                            bbox=cand["bbox"],
                            confidence=cand["conf"],
                            frame_idx=frame_idx,
                            timestamp=ts_sec
                        )
                        is_violation = ("NO-Hardhat" in cand["finding"].get("observed_evidence", "") or
                                        "NO-Safety Vest" in cand["finding"].get("observed_evidence", "") or
                                        "Violation" in cand["finding"].get("label", ""))
                        if is_violation:
                            new_track.ppe_violation_count += 1
                        cand["finding"]["track_id"] = new_track.track_id
                        cand["finding"]["frame_timestamp"] = ts_sec
                        
                        active_tracks.append(new_track)
                        all_tracks.append(new_track)
                        next_track_id += 1

                # Filter out lost tracks
                active_tracks = [t for t in active_tracks if t.missing_streak <= self.max_lost_frames]

                # Annotate the frame with bounding boxes and track IDs
                annotated_frame = frame.copy()
                for f in frame_findings:
                    bbox = f.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    t_id = f.get("track_id")
                    t_str = f" [Track #{t_id}]" if t_id else ""
                    lbl = f.get("label", "Detection")
                    
                    if "Compliant" in lbl or "Hardhat" in lbl:
                        color = (40, 167, 69)
                    elif "Violation" in lbl or "NO-" in lbl or "Defect" in lbl:
                        color = (30, 30, 220)
                    else:
                        color = (176, 108, 43)

                    draw_styled_box(
                        annotated_frame,
                        bbox,
                        label=f"{f.get('finding_uid', 'F')}: {lbl}{t_str}",
                        color=color,
                        thickness=2
                    )

                ann_frame_filename = f"frame_{idx:04d}.jpg"
                ann_frame_path = os.path.join(video_ann_dir, ann_frame_filename)
                cv2.imwrite(ann_frame_path, annotated_frame)
                ann_frame_url = f"/storage/annotations/video_{session_id[:8]}/{ann_frame_filename}"

                # Clean up temporary frame file
                if os.path.exists(temp_frame_path):
                    try:
                        os.remove(temp_frame_path)
                    except Exception:
                        pass

                all_session_findings.extend(frame_findings)
                frame_results.append({
                    "frame_index": frame_idx,
                    "sample_index": idx,
                    "timestamp_sec": ts_sec,
                    "findings_count": len(frame_findings),
                    "per_frame_counts": frame_per_class_counts,
                    "findings": frame_findings,
                    "annotated_frame_url": ann_frame_url
                })

        finally:
            cap.release()

        # 3. Calculate unique non-duplicate tracked entities
        unique_tracked_entities: Dict[str, int] = {}
        for track in all_tracks:
            unique_tracked_entities[track.class_label] = unique_tracked_entities.get(track.class_label, 0) + 1

        total_unique_entities = sum(unique_tracked_entities.values())

        # 4. Temporal Violation Persistence Reasoning
        # Rule: Do not claim a violation persists across time unless tracking evidence supports that conclusion!
        temporal_violations: List[Dict[str, Any]] = []
        for track in all_tracks:
            if track.ppe_violation_count > 0:
                is_persistent = (track.ppe_violation_count >= 2) or (track.frames_seen >= 2 and (track.ppe_violation_count / track.frames_seen) >= 0.5)
                status_cat = "Persistent Violation" if is_persistent else "Single-Frame Anomaly"
                severity = "CRITICAL" if is_persistent else "WARNING"
                evidence_note = (
                    f"Tracked entity observed across {track.frames_seen} frames with non-compliance detected in "
                    f"{track.ppe_violation_count} frames. Provenance: Temporal persistence verified across {track.last_time - track.start_time:.1f}s."
                    if is_persistent else
                    f"Non-compliance detected in only 1 frame ({track.ppe_violation_count}/{track.frames_seen} frames). "
                    f"Possible transient camera angle, occlusion, or motion artifact. Human review required before formal citation."
                )

                temporal_violations.append({
                    "track_id": track.track_id,
                    "class_label": track.class_label,
                    "frames_observed": track.frames_seen,
                    "violation_frames": track.ppe_violation_count,
                    "duration_sec": round(track.last_time - track.start_time, 2),
                    "persistence_category": status_cat,
                    "severity": severity,
                    "is_persistent": is_persistent,
                    "evidence_note": evidence_note
                })

        total_latency_ms = round((time.time() - t0) * 1000.0, 2)
        synthesis = (
            f"Processed {len(frame_results)} frames ({sample_fps} FPS) spanning {duration_sec}s. "
            f"Identified {total_unique_entities} unique non-duplicate entities across temporal tracks. "
            f"Found {len(temporal_violations)} safety compliance track incidents ({sum(1 for v in temporal_violations if v['is_persistent'])} verified persistent)."
        )

        return {
            "session_id": session_id,
            "video_path": video_path,
            "duration_sec": duration_sec,
            "total_video_frames": total_frames,
            "sampled_frames_processed": len(frame_results),
            "sample_fps": sample_fps,
            "unique_tracked_entities": unique_tracked_entities,
            "total_unique_entities": total_unique_entities,
            "frame_results": frame_results,
            "temporal_violations": temporal_violations,
            "detected_intent": detected_intent,
            "selected_agents": selected_agents,
            "synthesis_text": synthesis,
            "total_latency_ms": total_latency_ms
        }

video_tracking_service = VideoTrackingService()
