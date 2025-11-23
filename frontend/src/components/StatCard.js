import React from 'react';
import './StatCard.css';

const StatCard = ({ icon: Icon, label, value, color = 'blue', showProgress = false, trend }) => {
  const getColorClass = () => {
    const colorMap = {
      blue: 'stat-card-blue',
      green: 'stat-card-green',
      yellow: 'stat-card-yellow',
      orange: 'stat-card-orange',
      red: 'stat-card-red'
    };
    return colorMap[color] || colorMap.blue;
  };

  const getProgressColor = () => {
    if (typeof value === 'string' && value.includes('%')) {
      const numValue = parseFloat(value);
      if (numValue >= 85) return 'progress-high';
      if (numValue >= 70) return 'progress-medium';
      return 'progress-low';
    }
    return '';
  };

  const progressValue = typeof value === 'string' && value.includes('%') 
    ? parseFloat(value) 
    : null;

  return (
    <div className={`stat-card ${getColorClass()}`}>
      <div className="stat-card-icon">
        {Icon && <Icon />}
      </div>
      <div className="stat-card-content">
        <div className="stat-card-label">{label}</div>
        <div className="stat-card-value">{value}</div>
        {showProgress && progressValue !== null && (
          <div className="stat-card-progress">
            <div 
              className={`stat-card-progress-bar ${getProgressColor()}`}
              style={{ width: `${Math.min(progressValue, 100)}%` }}
            />
          </div>
        )}
        {trend && (
          <div className={`stat-card-trend ${trend > 0 ? 'trend-up' : 'trend-down'}`}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default StatCard;


