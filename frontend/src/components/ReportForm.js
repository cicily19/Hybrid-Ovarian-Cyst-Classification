import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaUpload, FaUser, FaIdCard, FaCalendarAlt, FaFileMedical, FaSpinner, FaCheckCircle, FaExclamationTriangle, FaTimes, FaImage } from 'react-icons/fa';
import VerificationReviewModal from './VerificationReviewModal';
import './ReportForm.css';

const ReportForm = () => {
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientId, setPatientId] = useState('');
  const [gender, setGender] = useState('');
  const [dateOfScan, setDateOfScan] = useState('');
  const [symptoms, setSymptoms] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [caseId, setCaseId] = useState(null);
  const [verificationScore, setVerificationScore] = useState(null);
  const [verificationStatus, setVerificationStatus] = useState(null);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [progressMessages, setProgressMessages] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  const getAuthToken = () => {
    return localStorage.getItem('access_token');
  };

  const runPrediction = async (caseIdToPredict, token) => {
    setProgressMessages([]);
    setLoading(true);
    
    try {
      await new Promise((resolve, reject) => {
        fetch(`http://localhost:8000/predict/${caseIdToPredict}/stream?include_shap=false&token=${encodeURIComponent(token)}`, {
          method: 'GET',
          headers: { 'Accept': 'text/event-stream' }
        })
        .then(response => {
          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          
          const readStream = () => {
            reader.read().then(({ done, value }) => {
              if (done) {
                if (!predictionResult) {
                  setErrorMsg('Stream ended unexpectedly');
                  setLoading(false);
                  reject(new Error('Stream ended unexpectedly'));
                }
                return;
              }
              
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';
              
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.slice(6));
                    setProgressMessages(prev => [...prev, data]);
                    
                    if (data.status === 'complete' && data.data) {
                      setPredictionResult(data.data);
                      setLoading(false);
                      resolve(data.data);
                      return;
                    } else if (data.status === 'error') {
                      setErrorMsg(data.message);
                      setLoading(false);
                      reject(new Error(data.message));
                      return;
                    }
                  } catch (e) {
                    console.error('Error parsing SSE data:', e);
                  }
                }
              }
              
              readStream();
            }).catch(error => {
              console.error('Error reading stream:', error);
              setErrorMsg(`Error reading progress stream: ${error.message}`);
              setLoading(false);
              reject(error);
            });
          };
          
          readStream();
        })
        .catch(error => {
          console.error('Network error during prediction:', error);
          setErrorMsg(`Failed to connect to server during prediction. Error: ${error.message}`);
          setLoading(false);
          reject(error);
        });
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : (typeof err === 'string' ? err : JSON.stringify(err));
      setErrorMsg(errorMessage || 'Error during prediction.');
      setLoading(false);
    }
  };

  const handleReviewConfirm = async (confirmed) => {
    setShowReviewModal(false);
    
    if (!confirmed) {
      setErrorMsg('Prediction cancelled. Please upload a valid ultrasound image.');
      setLoading(false);
      return;
    }

    const token = getAuthToken();
    if (!token || !caseId) {
      setErrorMsg('Session expired. Please login again.');
      navigate('/login');
      return;
    }

    await runPrediction(caseId, token);
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setImageFile(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreview(reader.result);
        };
        reader.readAsDataURL(file);
      } else {
        setErrorMsg('Please upload an image file.');
      }
    }
  };

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    
    if (!imageFile) {
      setErrorMsg('Please upload an ultrasound image.');
      return;
    }

    const token = getAuthToken();
    if (!token) {
      setErrorMsg('Please login first.');
      navigate('/login');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('patient_name', patientName);
      formData.append('patient_id', patientId);
      formData.append('age', patientAge);
      formData.append('gender', gender);
      formData.append('date_of_scan', dateOfScan);
      formData.append('symptoms', symptoms);

      let uploadRes;
      try {
        uploadRes = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        });
      } catch (networkError) {
        console.error('Network error:', networkError);
        throw new Error(`Failed to connect to server. Please ensure the backend is running on http://localhost:8000. Error: ${networkError.message}`);
      }

      if (!uploadRes.ok) {
        const errorData = await uploadRes.json();
        if (Array.isArray(errorData.detail)) {
          const errorMessages = errorData.detail.map(err => `${err.loc?.join('.')}: ${err.msg}`).join(', ');
          throw new Error(errorMessages || 'Upload failed');
        }
        throw new Error(errorData.detail || 'Upload failed');
      }

      const uploadData = await uploadRes.json();
      console.log('Upload response:', uploadData);
      
      if (!uploadData || typeof uploadData !== 'object') {
        throw new Error('Invalid upload response format');
      }
      
      setCaseId(uploadData.case_id);
      setVerificationScore(uploadData.verification_score);
      setVerificationStatus(uploadData.verification_status || (uploadData.verification_passed ? 'approved' : 'review_required'));

      if (uploadData.verification_status === 'rejected' || uploadData.verification_score < 0.50) {
        setErrorMsg(`Image verification failed. Verification score: ${uploadData.verification_score.toFixed(3)} (minimum: 0.50). The uploaded image does not appear to be a valid medical ultrasound image.`);
        setLoading(false);
        return;
      } else if (uploadData.verification_status === 'review_required' || (uploadData.verification_score >= 0.50 && uploadData.verification_score < 0.65)) {
        setShowReviewModal(true);
        setLoading(false);
        return;
      }

      await runPrediction(uploadData.case_id, token);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : (typeof err === 'string' ? err : JSON.stringify(err));
      setErrorMsg(errorMessage || 'Error processing request.');
      console.error('Error in handleSubmit:', err);
    } finally {
      setLoading(false);
    }
  };

  if (predictionResult && caseId) {
    return (
      <div className="result-container">
        <div className={`verification-badge ${verificationScore >= 0.70 ? 'success' : 'warning'}`}>
          {verificationScore >= 0.70 ? (
            <>
              <FaCheckCircle /> Ultrasound verification passed (Score: {verificationScore.toFixed(3)})
            </>
          ) : (
            <>
              <FaExclamationTriangle /> Ultrasound verification score: {verificationScore.toFixed(3)} (threshold: 0.70). Using heuristic verification.
            </>
          )}
        </div>

        <h3 className="result-title">Classification Results</h3>
        
        <div className="result-content">
          <div className="prediction-header">
            <div className="prediction-class">
              {predictionResult?.predicted_class || 'N/A'}
            </div>
            <div className="prediction-confidence">
              Confidence: {predictionResult?.confidence ? (predictionResult.confidence * 100).toFixed(1) : 'N/A'}%
            </div>
          </div>
          
          {predictionResult?.probabilities && (
            <div className="probability-bars">
              <div className="probability-item">
                <div className="probability-label">Probability (Simple):</div>
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar simple"
                    style={{ width: `${(predictionResult.probabilities.Simple * 100).toFixed(1)}%` }}
                  >
                    {(predictionResult.probabilities.Simple * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="probability-item">
                <div className="probability-label">Probability (Complex):</div>
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar complex"
                    style={{ width: `${(predictionResult.probabilities.Complex * 100).toFixed(1)}%` }}
                  >
                    {(predictionResult.probabilities.Complex * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {predictionResult.explanation && (
            <div className="explanation-box">
              <strong>Explanation:</strong> {predictionResult.explanation}
            </div>
          )}
        </div>

        <h4 className="visualization-title">Explainability Visualizations</h4>
        
        <div className="visualization-grid">
          {predictionResult.occlusion_heatmap && (
            <div className="visualization-item">
              <strong>Occlusion Heatmap:</strong>
              <img 
                src={`http://localhost:8000/static/shap/${predictionResult.occlusion_heatmap}`} 
                alt="Occlusion Heatmap" 
                className="visualization-image"
              />
            </div>
          )}
          {predictionResult.occlusion_overlay && (
            <div className="visualization-item">
              <strong>Occlusion Overlay:</strong>
              <img 
                src={`http://localhost:8000/static/shap/${predictionResult.occlusion_overlay}`} 
                alt="Occlusion Overlay" 
                className="visualization-image"
              />
            </div>
          )}
          {predictionResult.shap_heatmap && (
            <div className="visualization-item">
              <strong>SHAP Heatmap:</strong>
              <img 
                src={`http://localhost:8000/static/shap/${predictionResult.shap_heatmap}`} 
                alt="SHAP Heatmap" 
                className="visualization-image"
              />
            </div>
          )}
          {predictionResult.shap_overlay && (
            <div className="visualization-item">
              <strong>SHAP Overlay:</strong>
              <img 
                src={`http://localhost:8000/static/shap/${predictionResult.shap_overlay}`} 
                alt="SHAP Overlay" 
                className="visualization-image"
              />
            </div>
          )}
        </div>

        <div className="result-actions">
          <button 
            onClick={() => navigate(`/case/${caseId}`)}
            className="btn-primary"
          >
            View Full Case
          </button>
          <button 
            onClick={async () => {
              const token = getAuthToken();
              if (!token) {
                setErrorMsg('Please login first.');
                navigate('/login');
                return;
              }
              try {
                const response = await fetch(`http://localhost:8000/report/${caseId}`, {
                  headers: {
                    'Authorization': `Bearer ${token}`
                  }
                });
                if (!response.ok) {
                  if (response.status === 401) {
                    setErrorMsg('Authentication failed. Please login again.');
                    navigate('/login');
                    return;
                  }
                  throw new Error(`Failed to generate report: ${response.statusText}`);
                }
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `report_${caseId}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
              } catch (error) {
                setErrorMsg(`Error downloading report: ${error.message}`);
              }
            }}
            className="btn-secondary"
          >
            Download PDF Report
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="report-form-container">
      <div className="form-header">
        <h2 className="form-title">Upload New Case</h2>
        <p className="form-subtitle">Fill in patient information and upload ultrasound image for analysis</p>
      </div>

      <form onSubmit={handleSubmit} className="report-form" noValidate>
        <div className="form-section">
          <h3 className="section-title">Patient Information</h3>
          
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="patientName" className="form-label">
                <FaUser className="label-icon" />
                Patient Name
              </label>
              <input 
                type="text" 
                id="patientName"
                value={patientName} 
                onChange={e => setPatientName(e.target.value)} 
                required 
                className="form-input"
                placeholder="Enter patient name"
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="patientId" className="form-label">
                <FaIdCard className="label-icon" />
                Patient ID
              </label>
              <input 
                type="text" 
                id="patientId"
                value={patientId} 
                onChange={e => setPatientId(e.target.value)} 
                required 
                className="form-input"
                placeholder="Enter patient ID"
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="patientAge" className="form-label">
                Age
              </label>
              <input 
                type="number" 
                id="patientAge"
                value={patientAge} 
                onChange={e => setPatientAge(e.target.value)} 
                required 
                min="1"
                max="120"
                className="form-input"
                placeholder="Age"
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="gender" className="form-label">
                Gender
              </label>
              <select 
                id="gender"
                value={gender} 
                onChange={e => setGender(e.target.value)} 
                required 
                className="form-select"
                disabled={loading}
              >
                <option value="">Select gender...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
                <option value="Prefer not to say">Prefer not to say</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="dateOfScan" className="form-label">
              <FaCalendarAlt className="label-icon" />
              Date of Scan
            </label>
            <input 
              type="date" 
              id="dateOfScan"
              value={dateOfScan} 
              onChange={e => setDateOfScan(e.target.value)} 
              required 
              className="form-input"
              max={new Date().toISOString().split('T')[0]}
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="symptoms" className="form-label">
              <FaFileMedical className="label-icon" />
              Symptoms / Clinical Notes
            </label>
            <textarea 
              id="symptoms"
              value={symptoms} 
              onChange={e => setSymptoms(e.target.value)} 
              rows={4} 
              className="form-textarea"
              placeholder="Enter symptoms or clinical notes (optional)" 
              disabled={loading}
            />
          </div>
        </div>

        <div className="form-section">
          <h3 className="section-title">Ultrasound Image</h3>
          
          <div className="image-upload-section">
            {imagePreview ? (
              <div className="image-preview-container">
                <div className="image-preview-wrapper">
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                  <button
                    type="button"
                    onClick={removeImage}
                    className="remove-image-btn"
                    aria-label="Remove image"
                    disabled={loading}
                  >
                    <FaTimes />
                  </button>
                </div>
                <p className="image-preview-name">{imageFile?.name}</p>
              </div>
            ) : (
              <div
                className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  id="imageUpload"
                  accept="image/*"
                  onChange={handleImageChange}
                  required
                  className="file-input"
                  disabled={loading}
                />
                <label htmlFor="imageUpload" className="drop-zone-label">
                  <FaImage className="upload-icon" />
                  <span className="upload-text">
                    <strong>Click to upload</strong> or drag and drop
                  </span>
                  <span className="upload-hint">PNG, JPG, JPEG up to 10MB</span>
                </label>
              </div>
            )}
          </div>
        </div>
        
        {errorMsg && (
          <div className="error-message" role="alert">
            <FaExclamationTriangle className="error-icon" />
            <span>{errorMsg}</span>
            <button
              type="button"
              onClick={() => setErrorMsg('')}
              className="error-close-btn"
              aria-label="Close error"
            >
              <FaTimes />
            </button>
          </div>
        )}
        
        {loading && progressMessages.length > 0 && (
          <div className="progress-container">
            <div className="progress-header">
              <FaSpinner className="spinner-icon" />
              <span>Processing Status:</span>
            </div>
            <div className="progress-messages">
              {progressMessages.map((msg, index) => (
                <div 
                  key={index}
                  className={`progress-message ${msg.status || 'info'}`}
                >
                  <span className="progress-icon">
                    {msg.status === 'progress' || msg.status === 'loading' ? '⏳' :
                     msg.status === 'error' ? '❌' :
                     msg.status === 'warning' ? '⚠️' :
                     msg.status === 'complete' ? '✓' : '•'}
                  </span>
                  <span>{msg.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <button 
          type="submit" 
          disabled={loading}
          className={`submit-button ${loading ? 'loading' : ''}`}
        >
          {loading ? (
            <>
              <FaSpinner className="button-spinner" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <FaUpload />
              <span>Upload & Analyze</span>
            </>
          )}
        </button>
      </form>
      
      <VerificationReviewModal
        isOpen={showReviewModal}
        onClose={() => {
          setShowReviewModal(false);
          setLoading(false);
        }}
        onConfirm={handleReviewConfirm}
        verificationScore={verificationScore}
        imagePreview={imagePreview}
      />
    </div>
  );
};

export default ReportForm;
