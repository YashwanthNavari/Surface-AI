import React from 'react';
import { Sun, Moon, ShieldCheck } from 'lucide-react';

export default function Topbar({
  activeWorkspace,
  systemMode,
  setSystemMode,
  theme,
  toggleTheme,
}) {
  const titles = {
    mission: 'Overview',
    inspect: 'Live Inspection',
    explain: 'Explainability Lab',
    research: 'Research & Experiments',
    history: 'Audit History',
  };

  const currentTitle = titles[activeWorkspace] || 'Workspace';

  return (
    <header className="topbar">
      {/* Left: Clean Breadcrumb */}
      <div className="topbar-breadcrumbs">
        <span className="topbar-crumb-root">Platform</span>
        <span className="topbar-crumb-separator">/</span>
        <span className="topbar-crumb-current">{currentTitle}</span>
      </div>

      {/* Right: Mode & Theme Controls */}
      <div className="topbar-actions">
        {/* Operator vs Research Mode Segmented Switcher */}
        <div className="segmented-control">
          <button
            className={`segmented-btn ${systemMode === 'OPERATOR' ? 'active' : ''}`}
            onClick={() => setSystemMode('OPERATOR')}
            title="Streamlined for factory floor visual inspection"
          >
            Operator
          </button>
          <button
            className={`segmented-btn ${systemMode === 'RESEARCH' ? 'active' : ''}`}
            onClick={() => setSystemMode('RESEARCH')}
            title="Expanded academic telemetry and ablation analysis"
          >
            Research
          </button>
        </div>

        {/* Theme Toggle Button */}
        <button
          className="btn-secondary"
          onClick={toggleTheme}
          style={{ padding: '5px 10px', fontSize: '12px' }}
          title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
        >
          {theme === 'light' ? (
            <>
              <Moon className="w-3.5 h-3.5 text-slate-500" />
              <span>Dark</span>
            </>
          ) : (
            <>
              <Sun className="w-3.5 h-3.5 text-amber-500" />
              <span>Light</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}
