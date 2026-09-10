import React from 'react';
import { 
  Activity, 
  Cpu, 
  Layers, 
  Eye, 
  BarChart3, 
  History, 
  Radio, 
  CheckCircle2, 
  ShieldCheck, 
  Maximize2,
  Sun,
  Moon
} from 'lucide-react';

export default function Navigation({ 
  activeWorkspace, 
  setActiveWorkspace, 
  systemMode, 
  setSystemMode, 
  systemStatus, 
  totalInspections,
  theme,
  toggleTheme
}) {
  return (
    <header className="command-nav">
      {/* Brand & System Hierarchy */}
      <div className="brand-wrapper">
        <div className="brand-icon-box">
          <Cpu className="w-5 h-5 text-slate-900" />
        </div>
        <div>
          <div className="brand-text-main">SURFACE AI</div>
          <div className="brand-text-sub">INDUSTRIAL VISION PLATFORM & RESEARCH WORKSTATION</div>
        </div>
      </div>

      {/* 5 Primary Workspaces Switcher */}
      <nav className="workspace-tabs">
        <button
          className={`workspace-tab-btn ${activeWorkspace === 'mission' ? 'active' : ''}`}
          onClick={() => setActiveWorkspace('mission')}
        >
          <Radio className="w-3.5 h-3.5" />
          Mission Control
        </button>
        <button
          className={`workspace-tab-btn ${activeWorkspace === 'inspect' ? 'active' : ''}`}
          onClick={() => setActiveWorkspace('inspect')}
        >
          <Activity className="w-3.5 h-3.5" />
          Live Inspection
        </button>
        <button
          className={`workspace-tab-btn ${activeWorkspace === 'explain' ? 'active' : ''}`}
          onClick={() => setActiveWorkspace('explain')}
        >
          <Eye className="w-3.5 h-3.5" />
          Explainability Lab
        </button>
        <button
          className={`workspace-tab-btn ${activeWorkspace === 'research' ? 'active' : ''}`}
          onClick={() => setActiveWorkspace('research')}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          Research & Experiments
        </button>
        <button
          className={`workspace-tab-btn ${activeWorkspace === 'history' ? 'active' : ''}`}
          onClick={() => setActiveWorkspace('history')}
        >
          <History className="w-3.5 h-3.5" />
          Audit Stream
        </button>
      </nav>

      {/* Telemetry Cluster & Operator vs Research Mode */}
      <div className="telemetry-cluster">
        <div className="telemetry-pill">
          <span className="status-dot"></span>
          <span>ENGINE: {systemStatus}</span>
        </div>

        <div className="telemetry-pill">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          <span>3 DL MODELS</span>
        </div>

        {/* Mode Switcher */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${systemMode === 'OPERATOR' ? 'active' : ''}`}
            onClick={() => setSystemMode('OPERATOR')}
            title="Operator Mode: Streamlined for factory floor visual inspection"
          >
            OPERATOR
          </button>
          <button
            className={`mode-btn ${systemMode === 'RESEARCH' ? 'active' : ''}`}
            onClick={() => setSystemMode('RESEARCH')}
            title="Research Mode: Expanded academic metrics, ablation studies & training curves"
          >
            RESEARCH
          </button>
        </div>

        {/* Theme Mode Switcher */}
        <button
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
        >
          {theme === 'light' ? (
            <>
              <Moon className="w-3.5 h-3.5 text-slate-600" />
              <span>DARK</span>
            </>
          ) : (
            <>
              <Sun className="w-3.5 h-3.5 text-amber-400" />
              <span>LIGHT</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}
