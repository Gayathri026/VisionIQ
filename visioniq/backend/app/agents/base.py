"""
VisionIQ Specialist Agent Base Architecture
Provides strict typed interfaces, execution status enums, runtime measurements,
error isolation, and provenance tracking for all vision and cognitive agents.
"""

import time
import abc
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    SELECTED = "SELECTED"
    SKIPPED = "SKIPPED"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    TIMED_OUT = "TIMED_OUT"

class AgentInput(BaseModel):
    session_id: str = Field(..., description="Unique identifier of current analysis session")
    user_query: str = Field(..., description="Raw natural language prompt submitted by the operator")
    image_path: Optional[str] = Field(None, description="Absolute filesystem path to raw or sampled image")
    frame_timestamp: Optional[float] = Field(None, description="Video frame timestamp in seconds if applicable")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Shared upstream findings and context")
    timeout_seconds: float = Field(default=15.0, description="Maximum execution duration before isolating timeout")

class AgentOutput(BaseModel):
    agent_name: str = Field(..., description="Unique technical identifier of the specialist agent")
    status: ExecutionStatus = Field(..., description="Final execution lifecycle status")
    measured_runtime_ms: float = Field(0.0, description="Exact measured wall-clock execution duration in milliseconds")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Structured observations, detections, or text")
    data: Dict[str, Any] = Field(default_factory=dict, description="Specialist payload and raw metrics")
    error_details: Optional[str] = Field(None, description="Human-readable exception details if execution failed")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source tracking including model checkpoint, parameters, and deterministic rules"
    )

class BaseSpecialistAgent(abc.ABC):
    """
    Abstract Base Class for all VisionIQ specialist agents.
    Enforces deterministic error isolation, execution timing, and typed contracts.
    """

    def __init__(self, agent_name: str, model_version: str = "1.0.0"):
        self.agent_name = agent_name
        self.model_version = model_version

    @abc.abstractmethod
    def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
        """Core specialist logic implemented by subclasses."""
        pass

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Executes specialist logic with strict runtime measurement and error isolation.
        Guarantees that individual agent crashes never fail the pipeline.
        """
        start_time = time.time()
        try:
            output = self._execute_internal(agent_input)
            output.measured_runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            if not output.provenance:
                output.provenance = {
                    "agent": self.agent_name,
                    "version": self.model_version,
                    "timestamp": time.time()
                }
            return output
        except TimeoutError as te:
            runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.TIMED_OUT,
                measured_runtime_ms=runtime_ms,
                error_details=f"Agent timed out after {agent_input.timeout_seconds}s: {str(te)}",
                provenance={"agent": self.agent_name, "version": self.model_version}
            )
        except Exception as exc:
            runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.FAILED,
                measured_runtime_ms=runtime_ms,
                error_details=str(exc),
                provenance={"agent": self.agent_name, "version": self.model_version}
            )
