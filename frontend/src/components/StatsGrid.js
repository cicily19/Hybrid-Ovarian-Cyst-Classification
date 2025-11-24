import React from 'react';
import './StatsGrid.css';

const StatsGrid = ({ children }) => {
  return (
    <div className="stats-grid">
      {children}
    </div>
  );
};

export default StatsGrid;


