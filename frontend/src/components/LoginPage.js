import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import { Link } from 'react-router-dom';
import { FaEye, FaEyeSlash, FaSpinner } from 'react-icons/fa';
import '../styles/Auth.css';

const LoginPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false,
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({});
  const navigate = useNavigate();

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
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
    
    if (name === 'email' && value && !validateEmail(value)) {
      errors.email = 'Please enter a valid email address';
    } else if (name === 'password' && value && value.length < 6) {
      errors.password = 'Password must be at least 6 characters';
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
    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!validateEmail(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }
    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      errors.password = 'Password must be at least 6 characters';
    }

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
          rememberMe: formData.rememberMe,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        navigate('/dashboard');
      } else {
        setError(data.detail || 'Login failed. Please check your credentials and try again.');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError(`Unable to connect to server. Please ensure the backend is running on http://localhost:8000`);
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
        setError(data.detail || 'Google sign-in failed. Please try again.');
      }
    } catch (error) {
      setError('Unable to connect to server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleError = () => {
    setError('Google sign-in was cancelled or failed. Please try again.');
  };

  return (
    <GoogleOAuthProvider clientId="785766079315-nhad8c6l57rc47n82jdnn8kfjtrgrpjp.apps.googleusercontent.com">
      <div className="auth-container login-page">
        {/* Left side - Form */}
        <div className="form-side">
          <div className="form-wrapper">
            {/* Header */}
            <div className="header">
              <div className="header-top">
                <h1 className="title">Welcome back!</h1>
                <Link to="/signup" className="nav-link">
                  Sign up
                </Link>
              </div>
              <p className="subtitle">
                Enter your credentials to access your account
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
                    placeholder="Enter your password"
                    required
                    disabled={loading}
                    autoComplete="current-password"
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
                <div className="forgot-password">
                  <button
                    type="button"
                    className="forgot-link"
                    onClick={() => {
                      // TODO: Implement forgot password functionality
                      setError('Forgot password feature coming soon!');
                    }}
                  >
                    Forgot password?
                  </button>
                </div>
                {fieldErrors.password && (
                  <span className="field-error">{fieldErrors.password}</span>
                )}
              </div>

              {/* Remember me checkbox */}
              <div className="checkbox-group">
                <input
                  type="checkbox"
                  name="rememberMe"
                  checked={formData.rememberMe}
                  onChange={handleInputChange}
                  className="checkbox"
                  id="rememberMe"
                  disabled={loading}
                />
                <label htmlFor="rememberMe" className="checkbox-label">
                  Remember for 30 days
                </label>
              </div>

              {/* Submit button */}
              <button
                type="submit"
                className={`submit-button ${loading ? 'loading' : ''}`}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <FaSpinner className="spinner" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  'Sign in'
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
                Don't have an account?{' '}
                <Link to="/signup" className="footer-link">
                  Sign up
                </Link>
              </div>
            </form>
          </div>
        </div>

        {/* Right side - Image placeholder */}
        <div className="image-side">
          <div className="image-container login-bg">
            <div className="image-overlay">
              <div className="welcome-content">
                <h2>Ovarian Cyst Classification</h2>
                <p>Advanced AI-powered medical imaging analysis</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </GoogleOAuthProvider>
  );
};

export default LoginPage;
