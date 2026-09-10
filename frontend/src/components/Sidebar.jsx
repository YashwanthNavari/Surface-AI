import React from 'react';
import {
  LayoutDashboard,
  ScanEye,
  Eye,
  FlaskConical,
  History,
  ChevronLeft,
  ChevronRight,
  Activity,
  Cpu,
} from 'lucide-react';

export default function Sidebar({
  activeWorkspace,
  setActiveWorkspace,
  systemMode,
  systemStatus = 'ONLINE',
  collapsed,
  setCollapsed,
  totalInspections = 0,
}) {
  const navItems = [
    { id: 'mission', label: 'Overview', icon: LayoutDashboard },
    { id: 'inspect', label: 'Live Inspection', icon: ScanEye },
    { id: 'explain', label: 'Explainability', icon: Eye },
    // Show Research tab in Research Mode
    ...(systemMode === 'RESEARCH'
      ? [{ id: 'research', label: 'Research & Experiments', icon: FlaskConical }]
      : []),
    { id: 'history', label: 'Audit History', icon: History },
  ];

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Brand Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo-mark">
          <Cpu className="w-4 h-4" />
        </div>
        {!collapsed && (
          <>
            <span className="sidebar-brand-name">Surface AI</span>
            <span className="sidebar-version-tag">v1.2</span>
          </>
        )}
      </div>

      {/* Navigation Items */}
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeWorkspace === item.id;
          return (
            <button
              key={item.id}
              className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveWorkspace(item.id)}
              title={collapsed ? item.label : undefined}
            >
              <Icon />
              {!collapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer & Telemetry */}
      <div className="sidebar-footer">
        {!collapsed ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span className="status-dot" style={{ background: systemStatus === 'ONLINE' ? 'var(--accent-green)' : 'var(--accent-red)' }} />
                <span>Engine: {systemStatus}</span>
              </div>
              <span className="mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                {totalInspections} audits
              </span>
            </div>

            <button
              className="sidebar-collapse-btn"
              onClick={() => setCollapsed(true)}
              style={{ marginTop: '4px' }}
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              <span>Collapse sidebar</span>
            </button>
          </>
        ) : (
          <button
            className="sidebar-collapse-btn"
            onClick={() => setCollapsed(false)}
            title="Expand sidebar"
            style={{ justifyContent: 'center' }}
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </aside>
  );
}
