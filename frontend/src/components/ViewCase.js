import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';

const ViewCase = () => {
  const { caseId } = useParams();
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (!caseId) return;
    
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    fetchCase(caseId, token);
  }, [caseId, navigate]);

  const fetchCase = async (id, token) => {
    try {
      const response = await fetch(`http://localhost:8000/case/${id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem('access_token');
          navigate('/login');
          return;
        }
        if (response.status === 403) {
          setError('Not authorized to view this case');
          return;
        }
        if (response.status === 404) {
          setError('Case not found');
          return;
        }
        throw new Error('Failed to fetch case');
      }

      const data = await response.json();
      setCaseData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div style={{padding: '2rem', textAlign: 'center'}}>Loading case details...</div>;
  }

  if (error) {
    return (
      <div style={{maxWidth: '800px', margin: '2rem auto', padding: '2rem', background: '#fee2e2', color: '#991b1b', borderRadius: '0.5rem'}}>
        <p>{error}</p>
        <Link to="/history" style={{color: '#047857'}}>← Back to Case History</Link>
      </div>
    );
  }

  if (!caseData || !caseData.case) {
    return <div style={{padding: '2rem'}}>Case not found.</div>;
  }

  const { case: c, annotation } = caseData;

  return (
    <div style={{maxWidth: '1200px', margin: '2rem auto', padding: '0 1rem'}}>
      <div style={{marginBottom: '1rem'}}>
        <Link to="/history" style={{color: '#047857', textDecoration: 'none'}}>← Back to Case History</Link>
      </div>

      <div style={{background: '#fff', borderRadius: '0.5rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', padding: '2rem'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem'}}>
          <h2 style={{color: '#047857', margin: 0}}>Case Details</h2>
          <div style={{display: 'flex', gap: '1rem'}}>
            <Link 
              to={`/annotate/${c.id}`}
              style={{
                background: '#047857',
                color: '#fff',
                padding: '0.5rem 1rem',
                borderRadius: '0.25rem',
                textDecoration: 'none'
              }}
            >
              Add Annotation
            </Link>
            <button
              onClick={async () => {
                const token = localStorage.getItem('access_token');
                if (!token) {
                  navigate('/login');
                  return;
                }
                try {
                  const response = await fetch(`http://localhost:8000/report/${c.id}`, {
                    headers: {
                      'Authorization': `Bearer ${token}`
                    }
                  });
                  if (!response.ok) {
                    if (response.status === 401) {
                      navigate('/login');
                      return;
                    }
                    throw new Error(`Failed to generate report: ${response.statusText}`);
                  }
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `report_${c.id}.pdf`;
                  document.body.appendChild(a);
                  a.click();
                  window.URL.revokeObjectURL(url);
                  document.body.removeChild(a);
                } catch (error) {
                  setError(`Error downloading report: ${error.message}`);
                }
              }}
              style={{
                background: '#1f2937',
                color: '#fff',
                padding: '0.5rem 1rem',
                borderRadius: '0.25rem',
                border: 'none',
                cursor: 'pointer',
                textDecoration: 'none'
              }}
            >
              Download PDF
            </button>
          </div>
        </div>

        {/* Patient Information */}
        <div style={{marginBottom: '2rem'}}>
          <h3 style={{color: '#374151', marginBottom: '1rem', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem'}}>Patient Information</h3>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem'}}>
            <div>
              <strong style={{color: '#6b7280'}}>Patient Name:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem'}}>{c.patient_name}</div>
            </div>
            <div>
              <strong style={{color: '#6b7280'}}>Patient ID:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem'}}>{c.patient_id}</div>
            </div>
            <div>
              <strong style={{color: '#6b7280'}}>Age:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem'}}>{c.age || 'N/A'}</div>
            </div>
            <div>
              <strong style={{color: '#6b7280'}}>Gender:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem'}}>{c.gender || 'N/A'}</div>
            </div>
            <div>
              <strong style={{color: '#6b7280'}}>Date of Scan:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem'}}>
                {c.date_of_scan ? new Date(c.date_of_scan).toLocaleDateString() : 'N/A'}
              </div>
            </div>
            <div>
              <strong style={{color: '#6b7280'}}>Verification Score:</strong>
              <div style={{
                color: c.verification_score >= 0.70 ? '#047857' : '#dc2626',
                fontWeight: 'bold',
                marginTop: '0.25rem'
              }}>
                {c.verification_score !== null ? c.verification_score.toFixed(3) : 'N/A'}
                {c.verification_score !== null && c.verification_score < 0.70 && (
                  <span style={{fontSize: '0.875rem', color: '#dc2626', marginLeft: '0.5rem'}}>
                    (Below threshold)
                  </span>
                )}
              </div>
            </div>
          </div>
          {c.symptoms && (
            <div style={{marginTop: '1rem'}}>
              <strong style={{color: '#6b7280'}}>Clinical Notes:</strong>
              <div style={{color: '#374151', marginTop: '0.25rem', padding: '0.75rem', background: '#f3f4f6', borderRadius: '0.25rem'}}>
                {c.symptoms}
              </div>
            </div>
          )}
        </div>

        {/* Classification Results */}
        {c.predicted_class && (
          <div style={{marginBottom: '2rem'}}>
            <h3 style={{color: '#374151', marginBottom: '1rem', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem'}}>Classification Results</h3>
            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem'}}>
              <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
                <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.5rem'}}>Predicted Class</div>
                <div style={{fontSize: '1.25rem', fontWeight: 'bold', color: '#047857'}}>{c.predicted_class}</div>
              </div>
              <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
                <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.5rem'}}>P(Simple)</div>
                <div style={{fontSize: '1.25rem', fontWeight: 'bold', color: '#374151'}}>
                  {c.prob_simple !== null ? (c.prob_simple * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
                <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.5rem'}}>P(Complex)</div>
                <div style={{fontSize: '1.25rem', fontWeight: 'bold', color: '#374151'}}>
                  {c.prob_complex !== null ? (c.prob_complex * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Images */}
        <div style={{marginBottom: '2rem'}}>
          <h3 style={{color: '#374151', marginBottom: '1rem', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem'}}>Images</h3>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem'}}>
            {c.image_path && (
              <div>
                <strong style={{color: '#6b7280', display: 'block', marginBottom: '0.5rem'}}>Original Ultrasound</strong>
                <img 
                  src={`http://localhost:8000/${c.image_path}`} 
                  alt="Ultrasound" 
                  style={{maxWidth: '100%', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} 
                />
              </div>
            )}
            {c.shap_path && (
              <div>
                <strong style={{color: '#6b7280', display: 'block', marginBottom: '0.5rem'}}>Explainability Visualization</strong>
                <img 
                  src={`http://localhost:8000/${c.shap_path}`} 
                  alt="SHAP" 
                  style={{maxWidth: '100%', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} 
                />
              </div>
            )}
          </div>
        </div>

        {/* Radiologist Annotation */}
        {annotation && (
          <div style={{marginBottom: '2rem'}}>
            <h3 style={{color: '#374151', marginBottom: '1rem', borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem'}}>Radiologist Annotation</h3>
            <div style={{background: '#f3f4f6', padding: '1.5rem', borderRadius: '0.5rem'}}>
              <div style={{marginBottom: '1rem'}}>
                <strong style={{color: '#6b7280'}}>Radiologist:</strong>
                <div style={{color: '#374151', marginTop: '0.25rem'}}>{annotation.radiologist_name}</div>
              </div>
              {annotation.comments && (
                <div style={{marginBottom: '1rem'}}>
                  <strong style={{color: '#6b7280'}}>Comments:</strong>
                  <div style={{color: '#374151', marginTop: '0.25rem'}}>{annotation.comments}</div>
                </div>
              )}
              {annotation.severity && (
                <div style={{marginBottom: '1rem'}}>
                  <strong style={{color: '#6b7280'}}>Severity:</strong>
                  <div style={{color: '#374151', marginTop: '0.25rem'}}>{annotation.severity}</div>
                </div>
              )}
              {annotation.follow_up && (
                <div style={{marginBottom: '1rem'}}>
                  <strong style={{color: '#6b7280'}}>Follow-up:</strong>
                  <div style={{color: '#374151', marginTop: '0.25rem'}}>{annotation.follow_up}</div>
                </div>
              )}
              {annotation.created_at && (
                <div style={{fontSize: '0.875rem', color: '#6b7280'}}>
                  Created: {new Date(annotation.created_at).toLocaleString()}
                </div>
              )}
            </div>
          </div>
        )}

        {!annotation && (
          <div style={{textAlign: 'center', padding: '2rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
            <p style={{color: '#6b7280', marginBottom: '1rem'}}>No annotation available.</p>
            <Link 
              to={`/annotate/${c.id}`}
              style={{
                background: '#047857',
                color: '#fff',
                padding: '0.75rem 1.5rem',
                borderRadius: '0.25rem',
                textDecoration: 'none',
                display: 'inline-block'
              }}
            >
              Add Annotation
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default ViewCase;
