import React from 'react';

export default function ConfidenceBar({ value, isChampion = false }) {
  const percentage = Math.min(Math.max(value * 100, 0), 100);
  return (
    <div className="confidence-meter-bg">
      <div
        className={`confidence-meter-fill ${isChampion ? 'champion' : ''}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
