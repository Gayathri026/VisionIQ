import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { InspectionWorkspacePage } from './pages/InspectionWorkspacePage';
import { VideoTrackingPage } from './pages/VideoTrackingPage';
import { LiveCameraPage } from './pages/LiveCameraPage';
import { EvaluationDashboard } from './pages/EvaluationDashboard';
import { HistoryPage } from './pages/HistoryPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-[#0F1115] text-[#E2E8F0]">
        <Navbar />

        <main className="flex-1">
          <Routes>
            <Route path="/" element={<InspectionWorkspacePage />} />
            <Route path="/video" element={<VideoTrackingPage />} />
            <Route path="/camera" element={<LiveCameraPage />} />
            <Route path="/evaluation" element={<EvaluationDashboard />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>

        {/* Academic Evaluation Footer */}
        <footer className="border-t border-graphite-800 bg-graphite-950 py-4 px-6 text-center text-xs font-mono text-graphite-500">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-2">
            <div>
              <span>VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework</span>
              <span className="mx-2">•</span>
              <span>BE CSE Final Year Project (23CSP701)</span>
            </div>
            <div>
              <span>Dr. Mahalingam College of Engineering and Technology, Pollachi</span>
              <span className="mx-2">•</span>
              <span className="text-accent-cyan">IEEE Access Manuscript Ready</span>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
