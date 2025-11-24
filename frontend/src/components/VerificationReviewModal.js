import React from 'react';
import { FaExclamationTriangle, FaCheckCircle, FaTimes } from 'react-icons/fa';
import './VerificationReviewModal.css';

const VerificationReviewModal = ({ isOpen, onClose, onConfirm, verificationScore, imagePreview }) => {
  if (!isOpen) return null;

  const getScoreColor = (score) => {
    if (score < 0.50) return '#dc2626'; // Red
    if (score < 0.65) return '#f59e0b'; // Orange/Yellow
    return '#10b981'; // Green
  };

  const getScoreLabel = (score) => {
    if (score < 0.50) return 'Rejected';
    if (score < 0.65) return 'Review Required';
    return 'Approved';
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="verification-review-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-section">
            <FaExclamationTriangle className="modal-icon warning" />
            <h2>Image Verification Review Required</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close">
            <FaTimes />
          </button>
        </div>

        <div className="modal-content">
          <div className="verification-info">
            <p className="verification-message">
              The uploaded image has a verification score of <strong style={{ color: getScoreColor(verificationScore) }}>
                {(verificationScore * 100).toFixed(1)}%
              </strong>, which requires manual review.
            </p>
            
            <div className="score-breakdown">
              <div className="score-item">
                <span className="score-label">Verification Score:</span>
                <span className="score-value" style={{ color: getScoreColor(verificationScore) }}>
                  {(verificationScore * 100).toFixed(1)}%
                </span>
                <span className="score-status">{getScoreLabel(verificationScore)}</span>
              </div>
            </div>

            <div className="threshold-info">
              <div className="threshold-item">
                <span className="threshold-label">Below 50%:</span>
                <span className="threshold-desc">Rejected (Not ultrasound)</span>
              </div>
              <div className="threshold-item">
                <span className="threshold-label">50% - 65%:</span>
                <span className="threshold-desc">Review Required (Current)</span>
              </div>
              <div className="threshold-item">
                <span className="threshold-label">Above 65%:</span>
                <span className="threshold-desc">Auto-approved</span>
              </div>
            </div>

            {imagePreview && (
              <div className="image-preview-section">
                <p className="preview-label">Uploaded Image:</p>
                <img src={imagePreview} alt="Upload preview" className="preview-image" />
              </div>
            )}

            <div className="review-question">
              <p><strong>Please confirm:</strong> Is this a valid medical ultrasound image?</p>
            </div>
          </div>
        </div>

        <div className="modal-actions">
          <button className="btn-cancel" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-reject" onClick={() => onConfirm(false)}>
            Not an Ultrasound
          </button>
          <button className="btn-confirm" onClick={() => onConfirm(true)}>
            <FaCheckCircle /> Confirm & Continue
          </button>
        </div>
      </div>
    </div>
  );
};

export default VerificationReviewModal;


