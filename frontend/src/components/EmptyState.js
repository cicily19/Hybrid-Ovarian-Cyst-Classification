import React from 'react';
import './EmptyState.css';

const EmptyState = ({ icon: Icon, title, message, actionLabel, onAction }) => {
  return (
    <div className="empty-state">
      {Icon && (
        <div className="empty-state-icon">
          <Icon />
        </div>
      )}
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-message">{message}</p>
      {actionLabel && onAction && (
        <button className="empty-state-action" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
};

export default EmptyState;


