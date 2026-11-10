import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { AssistantDrawer } from './components/AssistantDrawer';
import { OverviewPage } from './pages/OverviewPage';
import { InspectionWorkspacePage } from './pages/InspectionWorkspacePage';
import { VideoTrackingPage } from './pages/VideoTrackingPage';
import { LiveCameraPage } from './pages/LiveCameraPage';
import { KnowledgeBasePage } from './pages/KnowledgeBasePage';
import { HistoryPage } from './pages/HistoryPage';
import { EvaluationDashboard } from './pages/EvaluationDashboard';
import { ResearchExperimentsPage } from './pages/ResearchExperimentsPage';
import { SettingsPage } from './pages/SettingsPage';
import { VisionIQAssistantPage } from './pages/VisionIQAssistantPage';

export const App: React.FC = () => {
  const [assistantDrawerOpen, setAssistantDrawerOpen] = useState(false);

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-[#0F1115] text-[#E2E8F0]">
        <Navbar onOpenAssistantDrawer={() => setAssistantDrawerOpen(true)} />

        <main className="flex-1">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/image" element={<InspectionWorkspacePage />} />
            <Route path="/video" element={<VideoTrackingPage />} />
            <Route path="/camera" element={<LiveCameraPage />} />
            <Route path="/knowledge" element={<KnowledgeBasePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/evaluation" element={<EvaluationDashboard />} />
            <Route path="/experiments" element={<ResearchExperimentsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/assistant" element={<VisionIQAssistantPage />} />
          </Routes>
        </main>

        {/* Global Floating AI Assistant Drawer */}
        <AssistantDrawer
          isOpen={assistantDrawerOpen}
          onClose={() => setAssistantDrawerOpen(false)}
        />

        {/* Academic Evaluation Footer */}
        <footer className="border-t border-graphite-800 bg-graphite-950 py-4 px-6 text-center text-xs font-mono text-graphite-500">
          <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row items-center justify-between gap-2">
            <div>
              <span>VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework</span>
              <span className="mx-2">•</span>
              <span>BE CSE Final Year Project (23CSP701)</span>
            </div>
            <div>
              <span>Dr. Mahalingam College of Engineering and Technology, Pollachi</span>
              <span className="mx-2">•</span>
              <span className="text-accent-cyan font-bold">IEEE Access Submission Ready</span>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
