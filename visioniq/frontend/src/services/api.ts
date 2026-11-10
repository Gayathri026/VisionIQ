import axios from 'axios';
import { AnalysisResponseData, BenchmarkSummary, HistorySession, AgentTrace, AssistantResponse, AssistantFAQItem, ResearchExperimentsData } from '../types';

const API_BASE = '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export const api = {
  checkHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  getOrchestrationPlan: async (query: string, weights?: { alpha?: number; beta?: number; gamma?: number; delta?: number; threshold?: number }) => {
    const res = await apiClient.post('/orchestration/plan', {
      query,
      ...weights,
    });
    return res.data;
  },

  analyzeImage: async (file: File, query: string): Promise<AnalysisResponseData> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('image', file);
    formData.append('query', query);

    const res = await apiClient.post('/analyze/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data.data || res.data;
  },

  getSessionTrace: async (sessionId: string): Promise<{ session_id: string; traces: AgentTrace[]; total_latency_ms: number }> => {
    const res = await apiClient.get(`/analysis/${sessionId}/trace`);
    return res.data;
  },

  processVideo: async (file: File, query: string, sampleFps: number = 1.0, maxFrames: number = 25) => {
    const formData = new FormData();
    formData.append('video', file);
    formData.append('query', query);
    formData.append('sample_fps', sampleFps.toString());
    formData.append('max_frames', maxFrames.toString());

    const res = await apiClient.post('/video/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data.data || res.data;
  },

  processCameraFrame: async (frameBase64: string, query: string, timestamp: number) => {
    const res = await apiClient.post('/camera/frame', {
      frame_base64: frameBase64,
      query,
      timestamp,
    });
    return res.data;
  },

  getHistory: async (
    paramsOrLimit?: number | { query?: string; input_type?: string; status?: string; limit?: number; order?: string }
  ): Promise<HistorySession[]> => {
    let params: any = { limit: 50 };
    if (typeof paramsOrLimit === 'number') {
      params = { limit: paramsOrLimit };
    } else if (paramsOrLimit) {
      params = paramsOrLimit;
    }
    const res = await apiClient.get('/history', { params });
    return res.data.sessions;
  },

  getSessionDetails: async (sessionId: string): Promise<AnalysisResponseData> => {
    const res = await apiClient.get(`/session/${sessionId}`);
    return res.data.session;
  },

  generatePdfReport: async (sessionId: string): Promise<{ status: string; report_uid: string; file_size_bytes: number; pdf_url: string }> => {
    const res = await apiClient.post(`/reports/${sessionId}/generate`);
    return res.data;
  },

  getReportMetadata: async (sessionId: string) => {
    const res = await apiClient.get(`/reports/${sessionId}/metadata`);
    return res.data;
  },

  runBenchmark: async (): Promise<BenchmarkSummary> => {
    const res = await apiClient.post('/evaluation/run');
    return res.data.results;
  },

  getBenchmarkRuns: async () => {
    const res = await apiClient.get('/evaluation/runs');
    return res.data.runs;
  },

  getRagDocuments: async () => {
    const res = await apiClient.get('/rag/documents');
    return res.data.documents;
  },

  ingestDocument: async (file: File, title?: string, sourceUri?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);
    if (sourceUri) formData.append('source_uri', sourceUri);
    const res = await apiClient.post('/rag/ingest', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  queryRag: async (query: string, topK: number = 3, threshold: number = 0.3) => {
    const res = await apiClient.post('/rag/query', {
      query,
      top_k: topK,
      relevance_threshold: threshold,
    });
    return res.data;
  },

  getPdfReportUrl: (sessionId: string) => {
    return `${API_BASE}/reports/${sessionId}/pdf`;
  },

  queryAssistant: async (query: string, sessionId?: string, currentTrace?: any[], activeFindingUid?: string): Promise<AssistantResponse> => {
    const res = await apiClient.post('/assistant/query', {
      query,
      session_id: sessionId,
      current_trace: currentTrace,
      active_finding_uid: activeFindingUid,
    });
    return res.data;
  },

  getAssistantFaqs: async (): Promise<AssistantFAQItem[]> => {
    const res = await apiClient.get('/assistant/faq');
    return res.data.questions;
  },

  getExperiments: async (): Promise<ResearchExperimentsData> => {
    const res = await apiClient.get('/experiments');
    return res.data;
  }
};
