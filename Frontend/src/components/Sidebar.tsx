
import React, { useState } from 'react';
import { LayoutDashboard, Users, Menu, ChevronLeft, BarChart } from 'lucide-react';
import { Link } from 'react-router-dom';

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  const toggleSidebar = () => {
    setCollapsed(!collapsed);
  };

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        {collapsed ? (
          <button onClick={toggleSidebar} className="btn btn-secondary btn-sm">
            <Menu size={20} />
          </button>
        ) : (
          <>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#2563EB" strokeWidth="2" />
              <path d="M16 10L12 6L8 10" stroke="#2563EB" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M12 18V8.5" stroke="#2563EB" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <h1>LeadScore</h1>
            <button onClick={toggleSidebar} className="btn btn-secondary btn-sm ml-auto">
              <ChevronLeft size={20} />
            </button>
          </>
        )}
      </div>

      <nav className={`sidebar-nav ${collapsed ? 'sidebar-collapsed-content' : ''}`}>
        <Link to="/" className="sidebar-link active">
          <LayoutDashboard className="sidebar-link-icon" />
          {!collapsed && <span className="sidebar-link-text">Dashboard</span>}
        </Link>
        <Link to="/leads" className="sidebar-link">
          <Users className="sidebar-link-icon" />
          {!collapsed && <span className="sidebar-link-text">Leads</span>}
        </Link>
        <Link to="/predictions" className="sidebar-link">
          <BarChart className="sidebar-link-icon" />
          {!collapsed && <span className="sidebar-link-text">Predictions</span>}
        </Link>
      </nav>

      {/* Mobile menu toggle */}
      <button className="menu-toggle">
        <Menu size={24} />
      </button>
    </aside>
  );
};

export default Sidebar;
