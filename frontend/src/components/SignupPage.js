import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { FaEye, FaEyeSlash, FaSpinner, FaCheck, FaTimes } from 'react-icons/fa';
import '../styles/Auth.css';

const SignupPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    agreeToTerms: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({});
  const [passwordStrength, setPasswordStrength] = useState({
    score: 0,
    feedback: []
  });

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const checkPasswordStrength = (password) => {
    const checks = {
      length: password.length >= 8,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      number: /[0-9]/.test(password),
      special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
    };

    const score = Object.values(checks).filter(Boolean).length;
    const feedback = [];
    
    if (!checks.length) feedback.push('At least 8 characters');
    if (!checks.uppercase) feedback.push('One uppercase letter');
    if (!checks.lowercase) feedback.push('One lowercase letter');
    if (!checks.number) feedback.push('One number');
    if (!checks.special) feedback.push('One special character');

    return { score, feedback, checks };
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));

    // Check password strength
    if (name === 'password') {
      const strength = checkPasswordStrength(value);
      setPasswordStrength(strength);
    }

    // Clear errors when user starts typing
    if (error) setError('');
    if (fieldErrors[name]) {
      setFieldErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    const errors = { ...fieldErrors };

    if (name === 'name' && value && value.trim().length < 2) {
      errors.name = 'Name must be at least 2 characters';
    } else if (name === 'email' && value && !validateEmail(value)) {
      errors.email = 'Please enter a valid email address';
    } else if (name === 'password' && value) {
      const strength = checkPasswordStrength(value);
      if (strength.score < 3) {
        errors.password = 'Password is too weak';
      }
    } else {
      delete errors[name];
    }

    setFieldErrors(errors);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setFieldErrors({});

    // Validate form
    const errors = {};
    if (!formData.name || formData.name.trim().length < 2) {
      errors.name = 'Name must be at least 2 characters';
    }
    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!validateEmail(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }
    if (!formData.password) {
      errors.password = 'Password is required';
    } else {
      const strength = checkPasswordStrength(formData.password);
      if (strength.score < 3) {
        errors.password = 'Password is too weak. Please use a stronger password.';
      }
    }
    if (!formData.agreeToTerms) {
      errors.agreeToTerms = 'You must agree to the terms & policy';
    }

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          agreeToTerms: formData.agreeToTerms,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Signup failed. Please check your information and try again.');
      }
    } catch (error) {
      console.error('Signup error:', error);
      setError('Unable to connect to server. Please ensure the backend is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('http://localhost:8000/google-auth', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ credential: credentialResponse.credential }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Google sign-up failed. Please try again.');
      }
    } catch (error) {
      setError('Unable to connect to server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleError = () => {
    setError('Google sign-up was cancelled or failed. Please try again.');
  };

  const getPasswordStrengthLabel = (score) => {
    if (score === 0) return { text: '', color: '' };
    if (score <= 2) return { text: 'Weak', color: '#ef4444' };
    if (score <= 3) return { text: 'Fair', color: '#f59e0b' };
    if (score <= 4) return { text: 'Good', color: '#3b82f6' };
    return { text: 'Strong', color: '#10b981' };
  };

  const strengthLabel = getPasswordStrengthLabel(passwordStrength.score);

  return (
    <GoogleOAuthProvider clientId="785766079315-nhad8c6l57rc47n82jdnn8kfjtrgrpjp.apps.googleusercontent.com">
      <div className="auth-container signup-page">
        {/* Left side - Form */}
        <div className="form-side">
          <div className="form-wrapper">
            {/* Header */}
            <div className="header">
              <div className="header-top">
                <h1 className="title">Get Started Now</h1>
                <Link to="/login" className="nav-link">
                  Sign in
                </Link>
              </div>
              <p className="subtitle">
                Create an account to start using our platform
              </p>
            </div>

            {/* Error message */}
            {error && (
              <div className="error-message" role="alert">
                <span className="error-icon">⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {/* Form */}
            <form className="form" onSubmit={handleSubmit} noValidate>
              {/* Name field */}
              <div className="form-group">
                <label className="label" htmlFor="name">
                  Full Name
                </label>
                <div className="input-wrapper">
                  <input
                    type="text"
                    name="name"
                    id="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={`input ${fieldErrors.name ? 'input-error' : ''}`}
                    placeholder="Enter your full name"
                    required
                    disabled={loading}
                    autoComplete="name"
                  />
                </div>
                {fieldErrors.name && (
                  <span className="field-error">{fieldErrors.name}</span>
                )}
              </div>

              {/* Email field */}
              <div className="form-group">
                <label className="label" htmlFor="email">
                  Email address
                </label>
                <div className="input-wrapper">
                  <input
                    type="email"
                    name="email"
                    id="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={`input ${fieldErrors.email ? 'input-error' : ''}`}
                    placeholder="Enter your email"
                    required
                    disabled={loading}
                    autoComplete="email"
                  />
                </div>
                {fieldErrors.email && (
                  <span className="field-error">{fieldErrors.email}</span>
                )}
              </div>

              {/* Password field */}
              <div className="form-group">
                <label className="label" htmlFor="password">
                  Password
                </label>
                <div className="input-wrapper password-wrapper">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    id="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={`input ${fieldErrors.password ? 'input-error' : ''}`}
                    placeholder="Create a strong password"
                    required
                    disabled={loading}
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    tabIndex={0}
                  >
                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                  </button>
                </div>
                
                {/* Password strength indicator */}
                {formData.password && (
                  <div className="password-strength">
                    <div className="password-strength-bar">
                      <div
                        className="password-strength-fill"
                        style={{
                          width: `${(passwordStrength.score / 5) * 100}%`,
                          backgroundColor: strengthLabel.color
                        }}
                      />
                    </div>
                    {strengthLabel.text && (
                      <span
                        className="password-strength-label"
                        style={{ color: strengthLabel.color }}
                      >
                        {strengthLabel.text}
                      </span>
                    )}
                  </div>
                )}

                {/* Password requirements */}
                {formData.password && passwordStrength.feedback.length > 0 && (
                  <div className="password-requirements">
                    {passwordStrength.feedback.map((req, index) => (
                      <div key={index} className="requirement-item">
                        <span className="requirement-icon">
                          {passwordStrength.checks[Object.keys(passwordStrength.checks)[index]] ? (
                            <FaCheck className="check-icon" />
                          ) : (
                            <FaTimes className="times-icon" />
                          )}
                        </span>
                        <span>{req}</span>
                      </div>
                    ))}
                  </div>
                )}

                {fieldErrors.password && (
                  <span className="field-error">{fieldErrors.password}</span>
                )}
              </div>

              {/* Terms checkbox */}
              <div className="checkbox-group">
                <input
                  type="checkbox"
                  name="agreeToTerms"
                  checked={formData.agreeToTerms}
                  onChange={handleInputChange}
                  className="checkbox"
                  id="agreeToTerms"
                  required
                  disabled={loading}
                />
                <label htmlFor="agreeToTerms" className="checkbox-label">
                  I agree to the{' '}
                  <Link to="/terms" className="terms-link">
                    terms & policy
                  </Link>
                </label>
              </div>
              {fieldErrors.agreeToTerms && (
                <span className="field-error">{fieldErrors.agreeToTerms}</span>
              )}

              {/* Submit button */}
              <button
                type="submit"
                className={`submit-button ${loading ? 'loading' : ''}`}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <FaSpinner className="spinner" />
                    <span>Creating account...</span>
                  </>
                ) : (
                  'Create account'
                )}
              </button>

              {/* Divider */}
              <div className="divider">
                <div className="divider-line">
                  <div className="divider-border" />
                </div>
                <div className="divider-text">
                  <span>Or continue with</span>
                </div>
              </div>

              {/* Social login buttons */}
              <div className="social-buttons">
                <div className="google-login-wrapper">
                  <GoogleLogin
                    onSuccess={handleGoogleSuccess}
                    onError={handleGoogleError}
                    disabled={loading}
                    useOneTap
                  />
                </div>
              </div>

              {/* Footer text */}
              <div className="footer-text">
                Already have an account?{' '}
                <Link to="/login" className="footer-link">
                  Sign in
                </Link>
              </div>
            </form>
          </div>
        </div>

        {/* Right side - Image placeholder */}
        <div className="image-side">
          <div className="image-container signup-bg">
            <div className="image-overlay">
              <div className="welcome-content">
                <h2>Join Our Platform</h2>
                <p>Start analyzing medical images with AI-powered precision</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </GoogleOAuthProvider>
  );
};

export default SignupPage;
