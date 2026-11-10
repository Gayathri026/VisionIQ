import React from 'react';
import { NavLink } from 'react-router-dom';
import { ShieldCheck, Video, Camera, BarChart3, History, Cpu } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <header className="border-b border-graphite-700 bg-graphite-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-blue to-accent-cyan flex items-center justify-center text-white shadow-md shadow-accent-blue/20">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">VisionIQ</span>
              <span className="text-[10px] font-mono uppercase bg-accent-blue/20 text-accent-cyan px-2 py-0.5 rounded border border-accent-blue/30">
                Adaptive Multi-Agent v1.0
              </span>
            </div>
            <p className="text-xs text-graphite-400 font-mono">Cost-Aware Query Routing & LangGraph Orchestration</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-graphite-700 text-white border border-graphite-600'
                  : 'text-graphite-300 hover:text-white hover:bg-graphite-800'
              }`
            }
          >
            <ShieldCheck className="w-4 h-4 text-accent-cyan" />
            <span>Inspection Workspace</span>
          </NavLink>

          <NavLink
            to="/video"
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-graphite-700 text-white border border-graphite-600'
                  : 'text-graphite-300 hover:text-white hover:bg-graphite-800'
              }`
            }
          >
            <Video className="w-4 h-4 text-accent-amber" />
            <span>Video Tracking</span>
          </NavLink>

          <NavLink
            to="/camera"
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-graphite-700 text-white border border-graphite-600'
                  : 'text-graphite-300 hover:text-white hover:bg-graphite-800'
              }`
            }
          >
            <Camera className="w-4 h-4 text-accent-emerald" />
            <span>Live Camera</span>
          </NavLink>

          <NavLink
            to="/evaluation"
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-graphite-700 text-white border border-graphite-600'
                  : 'text-graphite-300 hover:text-white hover:bg-graphite-800'
              }`
            }
          >
            <BarChart3 className="w-4 h-4 text-accent-cyan" />
            <span>Scientific Evaluation</span>
          </NavLink>

          <NavLink
            to="/history"
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                isActive
                  ? 'bg-graphite-700 text-white border border-graphite-600'
                  : 'text-graphite-300 hover:text-white hover:bg-graphite-800'
              }`
            }
          >
            <History className="w-4 h-4 text-graphite-400" />
            <span>History</span>
          </NavLink>
        </nav>
      </div>
    </header>
  );
};
