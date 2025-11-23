import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaUpload, FaCheckCircle, FaTimes, FaExclamationTriangle } from 'react-icons/fa';
import './BatchUpload.css';

const BatchUpload = () => {
  const [files, setFiles] = useState([]);
  const [metadata, setMetadata] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    
    // Initialize metadata for each file
    const newMetadata = selectedFiles.map((file, idx) => ({
      patient_name: '',
      patient_id: `P${String(idx + 1).padStart(3, '0')}`,
      age: '',
      gender: 'Female',
      date_of_scan: new Date().toISOString().split('T')[0],
      symptoms: ''
    }));
    setMetadata(newMetadata);
  };

  const updateMetadata = (index, field, value) => {
    const newMetadata = [...metadata];
    newMetadata[index][field] = value;
    setMetadata(newMetadata);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (files.length === 0) {
      setError('Please select at least one file');
      return;
    }

    // Validate all required fields
    for (let i = 0; i < metadata.length; i++) {
      if (!metadata[i].patient_name || !metadata[i].patient_id || !metadata[i].age || !metadata[i].date_of_scan) {
        setError(`Please fill in all required fields for file ${i + 1} (${files[i].name})`);
        return;
      }
    }

    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    setLoading(true);
    
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('metadata', JSON.stringify(metadata));

      // Note: When using FormData, don't set Content-Type manually - browser sets it automatically
      // But Authorization header should still work
      const response = await fetch('http://localhost:8000/batch/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
          // Don't set Content-Type - browser will set it automatically for FormData
        },
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const result = await response.json();
      setUploadResult(result);
      setError('');
    } catch (error) {
      setError(error.message || 'Upload failed. Please try again.');
      console.error('Upload error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="batch-upload-container">
      <div className="batch-upload-content">
        <h1>Batch Upload</h1>
        <p className="subtitle">Upload multiple ultrasound images at once</p>

        {!uploadResult ? (
          <form onSubmit={handleSubmit} className="batch-upload-form">
            {error && (
              <div className="error-message" role="alert">
                <FaExclamationTriangle className="error-icon" />
                <span>{error}</span>
                <button
                  type="button"
                  onClick={() => setError('')}
                  className="error-close-btn"
                  aria-label="Close error"
                >
                  <FaTimes />
                </button>
              </div>
            )}
            <div className="file-select-section">
              <label className="file-input-label">
                <FaUpload className="upload-icon" /> Select Multiple Images
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </label>
              {files.length > 0 && (
                <div className="files-list">
                  <p><strong>{files.length}</strong> file(s) selected</p>
                </div>
              )}
            </div>

            {files.length > 0 && (
              <div className="metadata-section">
                <h3>Patient Information</h3>
                <p className="metadata-note">Fill in patient details for each image</p>
                <div className="metadata-table-container">
                  <table className="metadata-table">
                    <thead>
                      <tr>
                        <th>File</th>
                        <th>Patient Name *</th>
                        <th>Patient ID *</th>
                        <th>Age *</th>
                        <th>Gender</th>
                        <th>Date of Scan *</th>
                        <th>Symptoms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {files.map((file, idx) => (
                        <tr key={idx}>
                          <td className="filename-cell">{file.name}</td>
                          <td>
                            <input
                              type="text"
                              value={metadata[idx]?.patient_name || ''}
                              onChange={(e) => updateMetadata(idx, 'patient_name', e.target.value)}
                              required
                              className="metadata-input"
                            />
                          </td>
                          <td>
                            <input
                              type="text"
                              value={metadata[idx]?.patient_id || ''}
                              onChange={(e) => updateMetadata(idx, 'patient_id', e.target.value)}
                              required
                              className="metadata-input"
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              value={metadata[idx]?.age || ''}
                              onChange={(e) => updateMetadata(idx, 'age', e.target.value)}
                              required
                              min="1"
                              className="metadata-input"
                            />
                          </td>
                          <td>
                            <select
                              value={metadata[idx]?.gender || 'Female'}
                              onChange={(e) => updateMetadata(idx, 'gender', e.target.value)}
                              className="metadata-select"
                            >
                              <option value="Male">Male</option>
                              <option value="Female">Female</option>
                              <option value="Other">Other</option>
                            </select>
                          </td>
                          <td>
                            <input
                              type="date"
                              value={metadata[idx]?.date_of_scan || ''}
                              onChange={(e) => updateMetadata(idx, 'date_of_scan', e.target.value)}
                              required
                              className="metadata-input"
                            />
                          </td>
                          <td>
                            <input
                              type="text"
                              value={metadata[idx]?.symptoms || ''}
                              onChange={(e) => updateMetadata(idx, 'symptoms', e.target.value)}
                              placeholder="Optional"
                              className="metadata-input"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <button type="submit" disabled={loading || files.length === 0} className="submit-btn">
              {loading ? 'Uploading...' : 'Upload Batch'}
            </button>
          </form>
        ) : (
          <div className="upload-result">
            <div className="result-header">
              <FaCheckCircle className="success-icon" />
              <h2>Upload Complete!</h2>
            </div>
            <div className="result-summary">
              <div className="summary-item">
                <span className="label">Total Files:</span>
                <span className="value">{uploadResult.total_files}</span>
              </div>
              <div className="summary-item approved">
                <span className="label">Approved:</span>
                <span className="value">{uploadResult.summary.approved}</span>
              </div>
              <div className="summary-item review">
                <span className="label">Review Required:</span>
                <span className="value">{uploadResult.summary.review_required}</span>
              </div>
              <div className="summary-item rejected">
                <span className="label">Rejected:</span>
                <span className="value">{uploadResult.summary.rejected}</span>
              </div>
              {uploadResult.summary.errors > 0 && (
                <div className="summary-item error">
                  <span className="label">Errors:</span>
                  <span className="value">{uploadResult.summary.errors}</span>
                </div>
              )}
            </div>

            <div className="actions">
              <button
                onClick={() => navigate(`/batch/${uploadResult.batch_id}`)}
                className="btn-primary"
              >
                View Batch Details
              </button>
              <button
                onClick={() => {
                  setUploadResult(null);
                  setFiles([]);
                  setMetadata([]);
                }}
                className="btn-secondary"
              >
                Upload Another Batch
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BatchUpload;

