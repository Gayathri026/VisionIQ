export interface Finding {
  finding_uid: string;
  agent_source: 'object' | 'ppe' | 'defect' | 'ocr' | 'scene' | string;
  label: string;
  bbox?: [number, number, number, number] | null;
  norm_bbox?: [number, number, number, number] | null;
  detection_confidence?: number | null;
  anomaly_score?: number | null;
  routing_score?: number | null;
  observed_evidence: string;
  root_cause_hypothesis?: string | null;
  alternative_explanations?: string | null;
  recommendation?: string | null;
  verification_procedure?: string | null;
  human_review_required: boolean;
  frame_timestamp?: number | null;
  track_id?: number | null;
  model_name?: string | null;
  model_version?: string | null;
  score_interpretation?: string | null;
  person_id?: string | null;
  is_confirmed_defect?: boolean | null;
  spatial_zone?: string | null;
  uncertainty_flag?: string | null;
}

export interface AgentTrace {
  agent_name: string;
  relevance_score: number;
  evidence_capability: number;
  normalized_cost: number;
  expected_latency: number;
  final_routing_score: number;
  decision: 'SELECTED' | 'SKIPPED' | 'MANDATORY_DEPENDENCY' | 'FALLBACK';
  selection_reason?: string;
  execution_status: 'SUCCESS' | 'FAILED' | 'SKIPPED' | 'PENDING';
  actual_latency_ms: number;
  error_message?: string | null;
}

export interface RAGReference {
  document_title: string;
  section_reference: string;
  excerpt: string;
  similarity_score: number;
  source_uri?: string | null;
}

export interface AnalysisResponseData {
  session_id: string;
  created_at: string;
  user_query: string;
  input_type: 'image' | 'video' | 'camera';
  input_path: string;
  input_url?: string;
  annotated_path?: string;
  annotated_url?: string;
  detected_intent?: string;
  intent_confidence: number;
  selected_agents: string[];
  executed_agents: string[];
  skipped_agents: string[];
  failed_agents: string[];
  status: 'completed' | 'partial' | 'failed';
  total_latency_ms: number;
  synthesis_text?: string;
  evidence_support_score: number;
  pdf_report_path?: string;
  pdf_url?: string;
  findings: Finding[];
  traces: AgentTrace[];
  rag_references?: RAGReference[];
  rag_sources?: RAGReference[];
}

export interface BenchmarkMetrics {
  name: string;
  total_queries: number;
  routing_precision: number;
  routing_recall: number;
  routing_f1: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  mean_agents_invoked: number;
  latency_reduction_pct: number;
  failure_recovery_rate: number;
}

export interface BenchmarkSummary {
  timestamp: string;
  total_test_queries: number;
  configurations: {
    fixed_pipeline: BenchmarkMetrics;
    rule_based_routing: BenchmarkMetrics;
    adaptive_visioniq: BenchmarkMetrics;
  };
  ablations: {
    no_cost_penalties: {
      name: string;
      routing_f1: number;
      median_latency_ms: number;
      mean_agents_invoked: number;
    };
    no_rag_grounding: {
      name: string;
      routing_f1: number;
      median_latency_ms: number;
      mean_agents_invoked: number;
    };
  };
  per_query_traces: Array<{
    query_id: string;
    query: string;
    detected_intent: string;
    selected_agents: string[];
    expected_agents: string[];
    precision: number;
    recall: number;
    f1: number;
    estimated_latency_ms: number;
  }>;
}

export interface HistorySession {
  id: string;
  created_at: string;
  user_query: string;
  input_type: string;
  detected_intent: string;
  status: string;
  total_latency_ms: number;
  evidence_support_score: number;
  findings_count?: number;
  pdf_report_path?: string;
}

export interface AssistantResponse {
  status: string;
  query: string;
  title: string;
  answer: string;
  source_references: string[];
  recommended_navigation?: {
    path: string;
    label: string;
  };
  suggested_followups: string[];
  mode: string;
}

export interface AssistantFAQItem {
  id: string;
  question: string;
  category: string;
}

export interface ExperimentComparison {
  pipeline_type: string;
  name: string;
  description: string;
  status: 'completed' | 'not_yet_run';
  note?: string;
  metrics?: any;
  theoretical_mean_agents?: number;
  theoretical_latency_ms?: number;
}

export interface ResearchExperimentsData {
  status: string;
  total_experiments_recorded: number;
  executed_runs: any[];
  pipeline_comparisons: ExperimentComparison[];
  active_models: any[];
  datasets: Array<{
    name: string;
    task: string;
    version: string;
    license: string;
    classes: string[];
    status: string;
  }>;
}
