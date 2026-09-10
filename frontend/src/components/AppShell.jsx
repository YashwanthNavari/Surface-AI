import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function AppShell({
  activeWorkspace,
  setActiveWorkspace,
  systemMode,
  setSystemMode,
  systemStatus,
  totalInspections,
  theme,
  toggleTheme,
  children,
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="app-shell">
      {/* Left Sidebar */}
      <Sidebar
        activeWorkspace={activeWorkspace}
        setActiveWorkspace={setActiveWorkspace}
        systemMode={systemMode}
        systemStatus={systemStatus}
        collapsed={collapsed}
        setCollapsed={setCollapsed}
        totalInspections={totalInspections}
      />

      {/* Main Workspace Area */}
      <div className="main-wrapper">
        <Topbar
          activeWorkspace={activeWorkspace}
          systemMode={systemMode}
          setSystemMode={setSystemMode}
          theme={theme}
          toggleTheme={toggleTheme}
        />

        {/* Scrollable Canvas */}
        <main className="content-canvas">
          <div className="content-inner">{children}</div>
        </main>
      </div>
    </div>
  );
}
