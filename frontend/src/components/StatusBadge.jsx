import React from 'react';
import { CheckCircle2, AlertTriangle, ShieldCheck, Activity } from 'lucide-react';

export default function StatusBadge({ type = 'neutral', icon = true, children }) {
  let badgeClass = 'badge-neutral';
  let IconComponent = null;

  if (type === 'success' || type === 'agreed') {
    badgeClass = 'badge-green';
    IconComponent = CheckCircle2;
  } else if (type === 'warning' || type === 'disagreed') {
    badgeClass = 'badge-amber';
    IconComponent = AlertTriangle;
  } else if (type === 'blue' || type === 'champion') {
    badgeClass = 'badge-blue';
    IconComponent = ShieldCheck;
  } else if (type === 'critical') {
    badgeClass = 'badge-red';
    IconComponent = AlertTriangle;
  } else if (type === 'online') {
    badgeClass = 'badge-green';
    IconComponent = Activity;
  }

  return (
    <span className={`badge ${badgeClass}`}>
      {icon && IconComponent && <IconComponent className="w-3.5 h-3.5" />}
      <span>{children}</span>
    </span>
  );
}
