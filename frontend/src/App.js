import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import SignupPage from './components/SignupPage';
import LoginPage from './components/LoginPage';
import Dashboard from './components/Dashboard';
import CaseHistory from './components/CaseHistory';
import ViewCase from './components/ViewCase';
import AnnotationForm from './components/AnnotationForm';
import InsightsDashboard from './components/InsightsDashboard';
import Profile from './components/Profile';
import NavBar from './components/NavBar';
import BatchUpload from './components/BatchUpload';
import BatchProcessing from './components/BatchProcessing';
import BatchList from './components/BatchList';

function App() {
  return (
    <Router>
      <div className="App">
        <NavBar />
        <Routes>
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<CaseHistory />} />
          <Route path="/case/:caseId" element={<ViewCase />} />
          <Route path="/annotate/:caseId" element={<AnnotationForm />} />
          <Route path="/insights" element={<InsightsDashboard />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/batch/upload" element={<BatchUpload />} />
          <Route path="/batch/:batchId" element={<BatchProcessing />} />
          <Route path="/batches" element={<BatchList />} />
          <Route path="/" element={<Navigate to="/signup" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;