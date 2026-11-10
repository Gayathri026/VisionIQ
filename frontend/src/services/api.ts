import axios from 'axios';
import { AnalysisResponseData, BenchmarkSummary, HistorySession } from '../types';

const API_BASE = '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minutes for heavy video processing
});

export const api = {
  checkHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  analyzeImage: async (file: File, query: string): Promise<AnalysisResponseData> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('query', query);

    const res = await apiClient.post('/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data.data;
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
    return res.data.data;
  },

  processCameraFrame: async (frameBase64: string, query: string, timestamp: number) => {
    const res = await apiClient.post('/camera/frame', {
      frame_base64: frameBase64,
      query,
      timestamp,
    });
    return res.data;
  },

  getHistory: async (limit: number = 50): Promise<HistorySession[]> => {
    const res = await apiClient.get(`/history?limit=${limit}`);
    return res.data.sessions;
  },

  getSessionDetails: async (sessionId: string): Promise<AnalysisResponseData> => {
    const res = await apiClient.get(`/session/${sessionId}`);
    return res.data.session;
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

  getPdfReportUrl: (sessionId: string) => {
    return `${API_BASE}/reports/${sessionId}/pdf`;
  }
};
