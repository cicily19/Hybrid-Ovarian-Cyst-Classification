
import React, { useState, useRef } from 'react';
import ReportForm from './ReportForm';

const OvarianClassifier = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showReportForm, setShowReportForm] = useState(false);
  const [occlusionHeatmap, setOcclusionHeatmap] = useState(null);
  const [occlusionOverlay, setOcclusionOverlay] = useState(null);
  const [shapHeatmap, setShapHeatmap] = useState(null);
  const [shapOverlay, setShapOverlay] = useState(null);
  const fileInputRef = useRef();

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImagePreview(ev.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Run prediction using backend
  const handlePredict = async () => {
    if (!fileInputRef.current || !fileInputRef.current.files[0]) {
      setPrediction(null);
      setConfidence(null);
      alert('Please upload an ultrasound image.');
      return;
    }
    setLoading(true);
    const file = fileInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setPrediction({
        predictedClass: data.predicted_class,
        explanation: `Simple: ${data.probabilities.Simple.toFixed(3)}, Complex: ${data.probabilities.Complex.toFixed(3)}`
      });
      setConfidence(data.confidence);
      setOcclusionHeatmap(data.occlusion_heatmap);
      setOcclusionOverlay(data.occlusion_overlay);
      setShapHeatmap(data.shap_heatmap);
      setShapOverlay(data.shap_overlay);
    } catch (err) {
      alert('Prediction failed.');
      setPrediction(null);
      setConfidence(null);
    }
    setLoading(false);
  };

  return (
    <div className="ovarian-classifier-panel">
  {/* Model performance section removed as requested */}
      <h3>Upload Ultrasound Image</h3>
      <input type="file" accept="image/*" ref={fileInputRef} onChange={handleImageUpload} />
      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="Preview" style={{ maxWidth: '200px', margin: '1rem 0' }} />
        </div>
      )}
      <button onClick={handlePredict} disabled={!imagePreview || loading} style={{ marginBottom: '1rem' }}>
        {loading ? 'Processing...' : 'Predict'}
      </button>
      {prediction && (
        <div className="prediction-result" style={{marginBottom: showReportForm ? '2rem' : '0'}}>
          <strong>Class:</strong> {prediction.predictedClass}<br />
          <strong>Confidence:</strong> {confidence}<br />
          <div className="prediction-explanation" style={{marginTop: '0.75rem', background: '#f3f4f6', padding: '0.75rem 1rem', borderRadius: '0.5rem', color: '#374151', fontSize: '1rem'}}>
            <strong>Explanation:</strong> {prediction.explanation}
          </div>
          {occlusionHeatmap && (
            <div style={{marginTop: '1rem'}}>
              <strong>Occlusion Heatmap:</strong><br />
              <img src={`http://localhost:8000/static/${occlusionHeatmap}`} alt="Occlusion Heatmap" style={{maxWidth: '300px', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} />
            </div>
          )}
          {occlusionOverlay && (
            <div style={{marginTop: '1rem'}}>
              <strong>Occlusion Overlay:</strong><br />
              <img src={`http://localhost:8000/static/${occlusionOverlay}`} alt="Occlusion Overlay" style={{maxWidth: '300px', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} />
            </div>
          )}
          {shapHeatmap && (
            <div style={{marginTop: '1rem'}}>
              <strong>SHAP Heatmap:</strong><br />
              <img src={`http://localhost:8000/static/${shapHeatmap}`} alt="SHAP Heatmap" style={{maxWidth: '300px', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} />
            </div>
          )}
          {shapOverlay && (
            <div style={{marginTop: '1rem'}}>
              <strong>SHAP Overlay:</strong><br />
              <img src={`http://localhost:8000/static/${shapOverlay}`} alt="SHAP Overlay" style={{maxWidth: '300px', borderRadius: '0.5rem', border: '1px solid #e5e7eb'}} />
            </div>
          )}
          {!showReportForm && (
            <button onClick={() => setShowReportForm(true)} style={{marginTop: '1rem', background: '#047857', color: '#fff', border: 'none', borderRadius: '0.3rem', padding: '0.75rem 1.5rem', cursor: 'pointer'}}>Generate Report</button>
          )}
        </div>
      )}
      {showReportForm && (
        <ReportForm prediction={prediction} confidence={confidence} imagePreview={imagePreview} />
      )}
    </div>
  );
};

export default OvarianClassifier;
