import React, { useState, useEffect, useCallback } from 'react';
import AppShell from './components/AppShell';
import MissionControl from './components/MissionControl';
import LiveInspection from './components/LiveInspection';
import ExplainabilityLab from './components/ExplainabilityLab';
import ResearchLab from './components/ResearchLab';
import HistoryStream from './components/HistoryStream';

export default function App() {
  const [activeWorkspace, setActiveWorkspace] = useState('inspect'); // Hero Live Inspection default
  const [systemMode, setSystemMode] = useState('OPERATOR'); // 'OPERATOR' | 'RESEARCH'
  const [inspectionResult, setInspectionResult] = useState(null);
  const [currentFile, setCurrentFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState('ONLINE');
  const [totalInspections, setTotalInspections] = useState(0);
  const [recentHistory, setRecentHistory] = useState([]);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('surface_ai_theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('surface_ai_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // Fetch backend status and history stats
  const fetchStatusAndHistory = useCallback(async () => {
    try {
      const resHealth = await fetch('http://127.0.0.1:8000/');
      if (resHealth.ok) {
        const data = await resHealth.json();
        setSystemStatus(data.status || 'ONLINE');
      } else {
        setSystemStatus('DEGRADED');
      }
    } catch {
      setSystemStatus('OFFLINE');
    }

    try {
      const resHist = await fetch('http://127.0.0.1:8000/api/v1/history?limit=15');
      if (resHist.ok) {
        const data = await resHist.json();
        setRecentHistory(data);
        setTotalInspections(data.length);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchStatusAndHistory();
    const interval = setInterval(fetchStatusAndHistory, 15000);
    return () => clearInterval(interval);
  }, [fetchStatusAndHistory]);

  const handleInspect = async (file) => {
    if (!file) return;
    setCurrentFile(file);
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference request failed: ${res.statusText}`);
      }

      const data = await res.json();
      setInspectionResult(data);
      setTotalInspections((prev) => prev + 1);
      fetchStatusAndHistory();
    } catch (err) {
      console.error('Inspection error:', err);
      alert('Failed to inspect specimen. Ensure FastAPI backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenGradcam = (file) => {
    if (file) {
      setCurrentFile(file);
    }
    setActiveWorkspace('explain');
  };

  const handleSampleSelect = async (defectClass, setPreviewCallback) => {
    try {
      setLoading(true);
      const res = await fetch('http://127.0.0.1:8000/api/v1/samples');
      const samples = await res.json();
      const match = samples.find((s) => s.class === defectClass) || samples[0];

      if (match) {
        const imgRes = await fetch(`http://127.0.0.1:8000${match.url}`);
        const blob = await imgRes.blob();
        const file = new File([blob], match.filename, { type: 'image/jpeg' });
        const previewUrl = URL.createObjectURL(blob);
        if (setPreviewCallback) {
          setPreviewCallback(file, previewUrl);
        }
        await handleInspect(file);
      }
    } catch (err) {
      console.error('Error fetching sample:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectHistoryItem = async (historyItem) => {
    try {
      setLoading(true);
      const res = await fetch('http://127.0.0.1:8000/api/v1/samples');
      const samples = await res.json();
      const match = samples.find((s) => s.class === historyItem.primary_prediction) || samples[0];
      if (match) {
        const imgRes = await fetch(`http://127.0.0.1:8000${match.url}`);
        const blob = await imgRes.blob();
        const file = new File([blob], historyItem.filename || match.filename, { type: 'image/jpeg' });
        setCurrentFile(file);
        await handleInspect(file);
      }
    } catch (err) {
      console.error('Error loading history image:', err);
    } finally {
      setLoading(false);
      setActiveWorkspace('inspect');
    }
  };

  return (
    <AppShell
      activeWorkspace={activeWorkspace}
      setActiveWorkspace={setActiveWorkspace}
      systemMode={systemMode}
      setSystemMode={setSystemMode}
      systemStatus={systemStatus}
      totalInspections={totalInspections}
      theme={theme}
      toggleTheme={toggleTheme}
    >
      {activeWorkspace === 'mission' && (
        <MissionControl
          onNavigateWorkspace={setActiveWorkspace}
          recentHistory={recentHistory}
        />
      )}

      {activeWorkspace === 'inspect' && (
        <LiveInspection
          onInspect={handleInspect}
          inspectionResult={inspectionResult}
          loading={loading}
          onOpenGradcam={handleOpenGradcam}
          onSampleSelect={handleSampleSelect}
          currentFile={currentFile}
        />
      )}

      {activeWorkspace === 'explain' && (
        <ExplainabilityLab
          currentFile={currentFile}
          onBackToInspection={() => setActiveWorkspace('inspect')}
        />
      )}

      {activeWorkspace === 'research' && (
        <ResearchLab />
      )}

      {activeWorkspace === 'history' && (
        <HistoryStream
          onSelectInspection={handleSelectHistoryItem}
        />
      )}
    </AppShell>
  );
}
