import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { 
  FaHome, 
  FaHistory, 
  FaChartLine, 
  FaUser, 
  FaSignOutAlt,
  FaBars,
  FaTimes,
  FaLayerGroup
} from 'react-icons/fa';
import './NavBar.css';

const NavBar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem('access_token');
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    navigate('/login');
  };

  if (!token) {
    return null; // Don't show navbar on auth pages
  }

  const isActive = (path) => location.pathname === path;

  const navLinks = [
    { path: '/dashboard', label: 'Dashboard', icon: FaHome },
    { path: '/history', label: 'Case History', icon: FaHistory },
    { path: '/batches', label: 'Batch Processing', icon: FaLayerGroup },
    { path: '/insights', label: 'Insights', icon: FaChartLine },
    { path: '/profile', label: 'Profile', icon: FaUser },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/dashboard" className="navbar-brand">
          <div className="navbar-logo">
            <FaChartLine className="logo-icon" />
            <span className="brand-text">Ovarian Cyst AI</span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <ul className="navbar-links">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <li key={link.path}>
                <Link 
                  to={link.path} 
                  className={`nav-link ${isActive(link.path) ? 'active' : ''}`}
                >
                  <Icon className="nav-icon" />
                  <span>{link.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>

        {/* Logout Button */}
        <div className="navbar-actions">
          <button 
            onClick={handleLogout}
            className="logout-btn"
            aria-label="Logout"
          >
            <FaSignOutAlt className="logout-icon" />
            <span className="logout-text">Logout</span>
          </button>
        </div>

        {/* Mobile Menu Toggle */}
        <button 
          className="mobile-menu-toggle"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <FaTimes /> : <FaBars />}
        </button>
      </div>

      {/* Mobile Navigation */}
      <div className={`mobile-nav ${mobileMenuOpen ? 'open' : ''}`}>
        <ul className="mobile-nav-links">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <li key={link.path}>
                <Link 
                  to={link.path} 
                  className={`mobile-nav-link ${isActive(link.path) ? 'active' : ''}`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <Icon className="mobile-nav-icon" />
                  <span>{link.label}</span>
                </Link>
              </li>
            );
          })}
          <li>
            <button 
              onClick={() => {
                handleLogout();
                setMobileMenuOpen(false);
              }}
              className="mobile-logout-btn"
            >
              <FaSignOutAlt className="mobile-logout-icon" />
              <span>Logout</span>
            </button>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default NavBar;
