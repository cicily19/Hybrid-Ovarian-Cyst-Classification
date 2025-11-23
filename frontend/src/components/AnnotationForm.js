import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const AnnotationForm = () => {
  const { caseId } = useParams();
  const [radiologistName, setRadiologistName] = useState('');
  const [comments, setComments] = useState('');
  const [severity, setSeverity] = useState('');
  const [followUp, setFollowUp] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
    }
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const payload = {
        case_id: parseInt(caseId),
        radiologist_name: radiologistName,
        comments,
        severity,
        follow_up: followUp
      };

      const response = await fetch('http://localhost:8000/annotate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to save annotation');
      }

      setSubmitted(true);
      setTimeout(() => {
        navigate(`/case/${caseId}`);
      }, 1500);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div style={{maxWidth: '800px', margin: '2rem auto', padding: '2rem', background: '#d1fae5', color: '#047857', borderRadius: '0.5rem', textAlign: 'center'}}>
        <div style={{fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '0.5rem'}}>✓ Annotation saved successfully!</div>
        <div>Redirecting to case details...</div>
      </div>
    );
  }

  return (
    <div style={{maxWidth: '800px', margin: '2rem auto', padding: '0 1rem'}}>
      <div style={{background: '#fff', borderRadius: '0.5rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', padding: '2rem'}}>
        <h2 style={{color: '#047857', marginBottom: '1.5rem'}}>Add Radiologist Annotation</h2>
        <p style={{color: '#6b7280', marginBottom: '1.5rem'}}>Add your professional assessment for Case #{caseId}</p>

        {error && (
          <div style={{background: '#fee2e2', color: '#991b1b', padding: '0.75rem', borderRadius: '0.5rem', marginBottom: '1rem'}}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div style={{marginBottom: '1.5rem'}}>
            <label style={{display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#374151'}}>
              Radiologist Name: *
            </label>
            <input 
              type="text" 
              value={radiologistName} 
              onChange={e => setRadiologistName(e.target.value)} 
              required 
              style={{
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '0.25rem',
                border: '1px solid #d1d5db',
                fontSize: '1rem'
              }} 
              placeholder="Enter your name"
            />
          </div>

          <div style={{marginBottom: '1.5rem'}}>
            <label style={{display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#374151'}}>
              Diagnostic Comments:
            </label>
            <textarea 
              value={comments} 
              onChange={e => setComments(e.target.value)} 
              rows={4} 
              style={{
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '0.25rem',
                border: '1px solid #d1d5db',
                fontSize: '1rem',
                fontFamily: 'inherit'
              }}
              placeholder="Enter your diagnostic comments and observations..."
            />
          </div>

          <div style={{marginBottom: '1.5rem'}}>
            <label style={{display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#374151'}}>
              Severity Assessment:
            </label>
            <select
              value={severity}
              onChange={e => setSeverity(e.target.value)}
              style={{
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '0.25rem',
                border: '1px solid #d1d5db',
                fontSize: '1rem'
              }}
            >
              <option value="">Select severity...</option>
              <option value="Low">Low</option>
              <option value="Moderate">Moderate</option>
              <option value="High">High</option>
              <option value="Critical">Critical</option>
            </select>
          </div>

          <div style={{marginBottom: '1.5rem'}}>
            <label style={{display: 'block', marginBottom: '0.5rem', fontWeight: 'bold', color: '#374151'}}>
              Follow-up Recommendation:
            </label>
            <textarea 
              value={followUp} 
              onChange={e => setFollowUp(e.target.value)} 
              rows={3} 
              style={{
                width: '100%', 
                padding: '0.75rem', 
                borderRadius: '0.25rem',
                border: '1px solid #d1d5db',
                fontSize: '1rem',
                fontFamily: 'inherit'
              }}
              placeholder="Enter follow-up recommendations..."
            />
          </div>

          <div style={{display: 'flex', gap: '1rem'}}>
            <button 
              type="submit" 
              disabled={loading}
              style={{
                background: loading ? '#9ca3af' : '#047857', 
                color: '#fff', 
                border: 'none', 
                borderRadius: '0.3rem', 
                padding: '0.75rem 2rem', 
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '1rem',
                fontWeight: 'bold'
              }}
            >
              {loading ? 'Saving...' : 'Save Annotation'}
            </button>
            <button 
              type="button"
              onClick={() => navigate(`/case/${caseId}`)}
              style={{
                background: '#6b7280', 
                color: '#fff', 
                border: 'none', 
                borderRadius: '0.3rem', 
                padding: '0.75rem 2rem', 
                cursor: 'pointer',
                fontSize: '1rem'
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AnnotationForm;
