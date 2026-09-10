import React from 'react';

export default function MetricCard({ label, value, subtext, highlightColor }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div
        className="kpi-value mono"
        style={highlightColor ? { color: highlightColor } : {}}
      >
        {value}
      </div>
      {subtext && <div className="kpi-subtext">{subtext}</div>}
    </div>
  );
}
