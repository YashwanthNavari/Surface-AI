import React from 'react';
import { ShieldCheck, Cpu, Activity, BarChart3, History, Eye, CheckCircle2, Sun, Moon } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, systemStatus, totalInspections, theme, toggleTheme }) {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon-wrapper">
          <Cpu className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="brand-title">SURFACE AI INSPECT</div>
          <div className="brand-subtitle">INDUSTRIAL DEFECT DETECTION & CLASSIFICATION</div>
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`nav-tab-btn ${activeTab === 'console' ? 'active' : ''}`}
          onClick={() => setActiveTab('console')}
        >
          <Activity className="w-4 h-4" />
          Live Inspection
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'explain' ? 'active' : ''}`}
          onClick={() => setActiveTab('explain')}
        >
          <Eye className="w-4 h-4" />
          Grad-CAM Studio
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          <BarChart3 className="w-4 h-4" />
          Research & Metrics
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <History className="w-4 h-4" />
          Audit History
        </button>
      </nav>

      <div className="telemetry-bar">
        <div className="telemetry-item">
          <span className="status-indicator"></span>
          <span>ENGINE: {systemStatus}</span>
        </div>
        <div className="telemetry-item">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <span>MODELS: 3 ACTIVE</span>
        </div>
        <div className="telemetry-item">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>INSPECTED: {totalInspections}</span>
        </div>

        {/* Theme Toggle Button */}
        <button
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Theme`}
          aria-label="Toggle Theme"
        >
          {theme === 'light' ? (
            <>
              <Moon className="w-3.5 h-3.5 text-slate-700" />
              <span>LIGHT</span>
            </>
          ) : (
            <>
              <Sun className="w-3.5 h-3.5 text-amber-400" />
              <span>DARK</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}
