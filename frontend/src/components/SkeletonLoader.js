import React from 'react';
import './SkeletonLoader.css';

const SkeletonLoader = ({ type = 'card', rows = 3, columns = 4 }) => {
  if (type === 'table') {
    return (
      <div className="skeleton-table">
        <div className="skeleton-table-header">
          {Array.from({ length: columns }).map((_, i) => (
            <div key={i} className="skeleton-cell skeleton-header" />
          ))}
        </div>
        {Array.from({ length: rows }).map((_, rowIdx) => (
          <div key={rowIdx} className="skeleton-table-row">
            {Array.from({ length: columns }).map((_, colIdx) => (
              <div key={colIdx} className="skeleton-cell" />
            ))}
          </div>
        ))}
      </div>
    );
  }

  if (type === 'card') {
    return (
      <div className="skeleton-card">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text short" />
      </div>
    );
  }

  if (type === 'grid') {
    return (
      <div className="skeleton-grid">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="skeleton-card">
            <div className="skeleton-circle" />
            <div className="skeleton-line skeleton-title" />
            <div className="skeleton-line skeleton-text" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="skeleton-loader">
      <div className="skeleton-line" />
      <div className="skeleton-line" />
      <div className="skeleton-line short" />
    </div>
  );
};

export default SkeletonLoader;


