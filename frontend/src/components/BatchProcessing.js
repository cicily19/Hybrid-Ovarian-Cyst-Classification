import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { FaPlay, FaCheckCircle, FaTimes, FaClock, FaExclamationTriangle } from 'react-icons/fa';
import './BatchProcessing.css';

const BatchProcessing = () => {
  const { batchId } = useParams();
  const [batch, setBatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({});
  const navigate = useNavigate();

  useEffect(() => {
    fetchBatchStatus();
    const interval = setInterval(fetchBatchStatus, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [batchId]);

  const fetchBatchStatus = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/batch/${batchId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setBatch(data);
        setLoading(false);
      }
    } catch (error) {
      console.error('Error fetching batch status:', error);
    }
  };

  const startProcessing = async () => {
    const token = localStorage.getItem('access_token');
    if (!token) return;

    setProcessing(true);
    
    try {
      const response = await fetch(
        `http://localhost:8000/batch/predict/${batchId}?auto_approve_review=false`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'text/event-stream'
          }
        }
      );

      if (!response.ok) {
        throw new Error('Failed to start batch processing');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.status === 'complete') {
                setProcessing(false);
                fetchBatchStatus();
              } else if (data.case_id) {
                setProgress(prev => ({
                  ...prev,
                  [data.case_id]: data
                }));
              }
            } catch (e) {
              console.error('Error parsing SSE:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error processing batch:', error);
      setProcessing(false);
      alert(`Error: ${error.message}`);
    }
  };

  if (loading) {
    return (
      <div className="batch-processing-container">
        <div className="loading-state">Loading batch details...</div>
      </div>
    );
  }

  if (!batch) {
    return (
      <div className="batch-processing-container">
        <div className="error-state">Batch not found</div>
      </div>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#10b981';
      case 'processing': return '#3b82f6';
      case 'uploaded': return '#f59e0b';
      case 'failed': return '#dc2626';
      default: return '#6b7280';
    }
  };

  const overallProgress = batch.total_cases > 0 
    ? Math.round((batch.completed_cases / batch.total_cases) * 100)
    : 0;

  return (
    <div className="batch-processing-container">
      <div className="batch-header">
        <div>
          <Link to="/batches" className="back-link">← Back to Batches</Link>
          <h1>{batch.batch_name}</h1>
        </div>
        <div className="batch-status">
          <span 
            className="status-badge" 
            style={{ backgroundColor: getStatusColor(batch.status) }}
          >
            {batch.status}
          </span>
        </div>
      </div>

      <div className="batch-summary">
        <div className="summary-card">
          <div className="summary-label">Total Cases</div>
          <div className="summary-value">{batch.total_cases}</div>
        </div>
        <div className="summary-card completed">
          <div className="summary-label">Completed</div>
          <div className="summary-value">{batch.completed_cases}</div>
        </div>
        <div className="summary-card failed">
          <div className="summary-label">Failed</div>
          <div className="summary-value">{batch.failed_cases}</div>
        </div>
        <div className="summary-card pending">
          <div className="summary-label">Pending</div>
          <div className="summary-value">{batch.pending_cases}</div>
        </div>
      </div>

      {batch.status === 'uploaded' && (
        <div className="action-section">
          <button 
            onClick={startProcessing} 
            disabled={processing} 
            className="start-btn"
          >
            <FaPlay /> {processing ? 'Processing...' : 'Start Processing'}
          </button>
        </div>
      )}

      {batch.status === 'processing' && (
        <div className="progress-section">
          <div className="progress-bar-container">
            <div className="progress-bar-label">
              Overall Progress: {overallProgress}%
            </div>
            <div className="progress-bar">
              <div 
                className="progress-bar-fill"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      <div className="cases-list">
        <h2>Cases ({batch.cases.length})</h2>
        <div className="table-container">
          <table className="cases-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Verification</th>
                <th>Status</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {batch.cases.map(caseItem => (
                <tr key={caseItem.case_id}>
                  <td>{caseItem.patient_name}</td>
                  <td>
                    <span className={`verification-badge ${
                      caseItem.verification_status === 'approved' ? 'approved' :
                      caseItem.verification_status === 'review_required' ? 'review' : 'rejected'
                    }`}>
                      {caseItem.verification_score ? (caseItem.verification_score * 100).toFixed(1) + '%' : 'N/A'}
                    </span>
                  </td>
                  <td>
                    <div className="status-cell">
                      {caseItem.status === 'completed' ? (
                        <FaCheckCircle className="status-icon completed" />
                      ) : caseItem.status === 'pending' ? (
                        <FaClock className="status-icon pending" />
                      ) : (
                        <FaTimes className="status-icon failed" />
                      )}
                      <span>{caseItem.status}</span>
                    </div>
                  </td>
                  <td>
                    {caseItem.predicted_class || (
                      caseItem.verification_status === 'review_required' ? (
                        <span className="review-required">Review Required</span>
                      ) : 'N/A'
                    )}
                  </td>
                  <td>
                    {caseItem.prob_simple !== null && caseItem.prob_complex !== null ? (
                      <span className="confidence-value">
                        {(Math.max(caseItem.prob_simple, caseItem.prob_complex) * 100).toFixed(1)}%
                      </span>
                    ) : 'N/A'}
                  </td>
                  <td>
                    <button 
                      onClick={() => navigate(`/case/${caseItem.case_id}`)}
                      className="view-btn"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BatchProcessing;


