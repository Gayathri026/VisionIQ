import os
import pytest
from backend.services.video_service import video_service

def test_video_processing():
    test_video = "sample_data/vidssave.com People Entering And Exiting Mall Stock Footage 720P.mp4"
    assert os.path.exists(test_video), f"Test video {test_video} missing"

    result = video_service.process_video(
        video_path=test_video,
        query="Track visible personnel and verify count",
        sample_fps=0.5,
        max_frames=4  # Fast test with 4 sampled frames
    )

    assert result["sampled_frames_processed"] > 0
    assert "unique_tracked_entities" in result
    assert result["duration_sec"] > 0.0
    assert len(result["frame_results"]) == result["sampled_frames_processed"]
