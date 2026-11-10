import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanEye,
  Video,
  Camera,
  BookOpen,
  History,
  BarChart3,
  FlaskConical,
  Settings,
  Bot,
  Menu,
  X,
  Cpu,
  Sparkles,
  Activity
} from 'lucide-react';

interface NavbarProps {
  onOpenAssistantDrawer?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onOpenAssistantDrawer }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { to: '/', label: 'Overview', icon: LayoutDashboard, exact: true },
    { to: '/image', label: 'Image Inspection', icon: ScanEye, exact: false },
    { to: '/video', label: 'Video Inspection', icon: Video, exact: false },
    { to: '/camera', label: 'Live Camera', icon: Camera, exact: false },
    { to: '/knowledge', label: 'Knowledge Base', icon: BookOpen, exact: false },
    { to: '/history', label: 'Analysis History', icon: History, exact: false },
    { to: '/evaluation', label: 'Model Evaluation', icon: BarChart3, exact: false },
    { to: '/experiments', label: 'Research Experiments', icon: FlaskConical, exact: false },
    { to: '/settings', label: 'Settings', icon: Settings, exact: false },
    { to: '/assistant', label: 'VisionIQ Assistant', icon: Bot, exact: false },
  ];

  return (
    <header className="border-b border-[#2B3A54] bg-[#10182B]/95 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3">
      <div className="max-w-[1760px] mx-auto flex items-center justify-between gap-4">
        {/* Brand Command Center Title */}
        <div className="flex items-center space-x-3.5 flex-shrink-0">
          <NavLink to="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#1E3A8A] via-[#0284C7] to-[#43D9F5] flex items-center justify-center text-white shadow-md shadow-accent-cyan/20 group-hover:scale-105 transition-transform">
              <Cpu className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <span className="font-extrabold text-xl tracking-tight text-white group-hover:text-[#43D9F5] transition-colors">
                  VisionIQ
                </span>
                <span className="text-[11px] font-mono uppercase bg-[#151F35] text-[#43D9F5] px-2 py-0.5 rounded-full border border-[#43D9F5]/40 font-bold shadow-sm">
                  Autonomous Command Center
                </span>
              </div>
              <p className="text-[12px] text-[#8998AF] font-sans hidden sm:block">
                Dynamic Orchestration • Cost-Aware Routing (Eq. 1) • Standards Grounding
              </p>
            </div>
          </NavLink>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="hidden 2xl:flex items-center space-x-1.5 flex-1 justify-center max-w-5xl">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.exact}
                className={({ isActive }) =>
                  `flex items-center space-x-2 px-3 py-2 rounded-lg text-[15px] font-medium transition-all ${
                    isActive
                      ? 'bg-[#1B2942] text-white border border-[#43D9F5]/50 shadow-sm shadow-[#43D9F5]/10 font-semibold'
                      : 'text-[#B4C0D4] hover:text-white hover:bg-[#151F35] border border-transparent'
                  }`
                }
              >
                <Icon className="w-4 h-4 text-[#43D9F5] flex-shrink-0" />
                <span className="whitespace-nowrap">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Right Actions: System Health Badge & Assistant Drawer Toggle */}
        <div className="flex items-center space-x-3 flex-shrink-0">
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-[#151F35] border border-[#2B3A54] text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-[#34D399] animate-pulse"></span>
            <span className="text-[#8998AF]">Engine:</span>
            <span className="text-[#F1F5F9] font-semibold">Active Ready</span>
          </div>

          {onOpenAssistantDrawer && (
            <button
              onClick={onOpenAssistantDrawer}
              className="hidden sm:flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-[#43D9F5]/15 hover:bg-[#43D9F5]/25 border border-[#43D9F5]/50 text-[#43D9F5] text-sm font-semibold transition-all shadow-sm shadow-[#43D9F5]/15 active:scale-95"
              title="Open Assistant Side Panel"
            >
              <Sparkles className="w-4 h-4 text-[#43D9F5] animate-pulse" />
              <span>AI Guide</span>
            </button>
          )}

          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="2xl:hidden p-2 rounded-lg bg-[#151F35] text-[#B4C0D4] hover:text-white border border-[#2B3A54]"
            aria-label="Toggle Navigation Menu"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile / Tablet Responsive Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="2xl:hidden border-t border-[#2B3A54] mt-3 pt-3 pb-2 px-2 bg-[#0B1020]/95 backdrop-blur-md rounded-b-xl shadow-xl">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.exact}
                  onClick={() => setMobileMenuOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center space-x-2.5 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-[#1B2942] text-white border border-[#43D9F5]'
                        : 'text-[#B4C0D4] hover:text-white hover:bg-[#151F35] border border-transparent'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 text-[#43D9F5] flex-shrink-0" />
                  <span className="truncate">{item.label}</span>
                </NavLink>
              );
            })}
          </div>

          {onOpenAssistantDrawer && (
            <button
              onClick={() => {
                setMobileMenuOpen(false);
                onOpenAssistantDrawer();
              }}
              className="mt-3 w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg bg-[#43D9F5]/20 border border-[#43D9F5]/50 text-[#43D9F5] text-sm font-semibold"
            >
              <Sparkles className="w-4 h-4" />
              <span>Open VisionIQ Assistant</span>
            </button>
          )}
        </div>
      )}
    </header>
  );
};
