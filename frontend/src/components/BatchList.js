import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FaPlus, FaClock, FaCheckCircle, FaTimes, FaSpinner } from 'react-icons/fa';
import './BatchList.css';

const BatchList = () => {
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    fetchBatches();
  }, []);

  const fetchBatches = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    try {
      const response = await fetch('http://localhost:8000/batches', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setBatches(data.batches);
        setTotal(data.total);
      }
    } catch (error) {
      console.error('Error fetching batches:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <FaCheckCircle className="status-icon completed" />;
      case 'processing':
        return <FaSpinner className="status-icon processing spinning" />;
      case 'uploaded':
        return <FaClock className="status-icon uploaded" />;
      case 'failed':
        return <FaTimes className="status-icon failed" />;
      default:
        return <FaClock className="status-icon" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#10b981';
      case 'processing': return '#3b82f6';
      case 'uploaded': return '#f59e0b';
      case 'failed': return '#dc2626';
      default: return '#6b7280';
    }
  };

  if (loading) {
    return (
      <div className="batch-list-container">
        <div className="loading-state">Loading batches...</div>
      </div>
    );
  }

  return (
    <div className="batch-list-container">
      <div className="batch-list-header">
        <div>
          <h1>Batch Processing</h1>
          <p className="subtitle">Manage and monitor batch uploads</p>
        </div>
        <Link to="/batch/upload" className="btn-primary">
          <FaPlus /> New Batch Upload
        </Link>
      </div>

      {batches.length === 0 ? (
        <div className="empty-state">
          <p>No batches yet. Create your first batch to get started.</p>
          <Link to="/batch/upload" className="btn-primary">
            <FaPlus /> Create Batch
          </Link>
        </div>
      ) : (
        <>
          <div className="batches-stats">
            <div className="stat-item">
              <span className="stat-label">Total Batches:</span>
              <span className="stat-value">{total}</span>
            </div>
          </div>

          <div className="batches-grid">
            {batches.map(batch => (
              <Link 
                key={batch.batch_id} 
                to={`/batch/${batch.batch_id}`} 
                className="batch-card"
              >
                <div className="batch-card-header">
                  <h3>{batch.batch_name}</h3>
                  <div className="batch-status-badge">
                    {getStatusIcon(batch.status)}
                    <span 
                      className="status-text"
                      style={{ color: getStatusColor(batch.status) }}
                    >
                      {batch.status}
                    </span>
                  </div>
                </div>
                <div className="batch-card-stats">
                  <div className="stat">
                    <span className="stat-label">Total:</span>
                    <span className="stat-value">{batch.total_cases}</span>
                  </div>
                  <div className="stat completed">
                    <span className="stat-label">Completed:</span>
                    <span className="stat-value">{batch.completed_cases}</span>
                  </div>
                  <div className="stat failed">
                    <span className="stat-label">Failed:</span>
                    <span className="stat-value">{batch.failed_cases}</span>
                  </div>
                </div>
                {batch.total_cases > 0 && (
                  <div className="batch-progress">
                    <div className="progress-bar-mini">
                      <div 
                        className="progress-fill-mini"
                        style={{ 
                          width: `${Math.round((batch.completed_cases / batch.total_cases) * 100)}%` 
                        }}
                      />
                    </div>
                    <span className="progress-text">
                      {Math.round((batch.completed_cases / batch.total_cases) * 100)}% Complete
                    </span>
                  </div>
                )}
                <div className="batch-card-footer">
                  <span className="date">
                    {new Date(batch.created_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default BatchList;


