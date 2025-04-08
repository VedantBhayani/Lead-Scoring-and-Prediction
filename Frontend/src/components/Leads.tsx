
import React, { useState } from 'react';
import { Search, Filter, ChevronDown, ChevronUp } from 'lucide-react';
import { Link } from 'react-router-dom';

const Leads = () => {
  // Sample leads data
  const initialLeads = [
    { id: 1, name: 'John Doe', email: 'john@example.com', phone: '(555) 123-4567', source: 'Website', score: 87, status: 'high' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', phone: '(555) 765-4321', source: 'Email', score: 65, status: 'medium' },
    { id: 3, name: 'Mike Johnson', email: 'mike@example.com', phone: '(555) 987-6543', source: 'Social', score: 42, status: 'low' },
    { id: 4, name: 'Sarah Williams', email: 'sarah@example.com', phone: '(555) 456-7890', source: 'Referral', score: 91, status: 'high' },
    { id: 5, name: 'Alex Brown', email: 'alex@example.com', phone: '(555) 234-5678', source: 'Website', score: 78, status: 'medium' },
    { id: 6, name: 'Emily Davis', email: 'emily@example.com', phone: '(555) 876-5432', source: 'Social', score: 53, status: 'medium' },
    { id: 7, name: 'David Wilson', email: 'david@example.com', phone: '(555) 345-6789', source: 'Email', score: 39, status: 'low' },
    { id: 8, name: 'Lisa Taylor', email: 'lisa@example.com', phone: '(555) 654-3210', source: 'Referral', score: 85, status: 'high' },
  ];

  const [leads, setLeads] = useState(initialLeads);
  const [sortColumn, setSortColumn] = useState('name');
  const [sortDirection, setSortDirection] = useState('asc');
  const [searchTerm, setSearchTerm] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Sorting logic
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }

    const sortedLeads = [...leads].sort((a: any, b: any) => {
      if (a[column] < b[column]) return sortDirection === 'asc' ? -1 : 1;
      if (a[column] > b[column]) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    setLeads(sortedLeads);
  };

  // Filter logic
  const applyFilters = () => {
    let filteredLeads = initialLeads;

    if (searchTerm) {
      filteredLeads = filteredLeads.filter(lead => 
        lead.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lead.email.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (sourceFilter) {
      filteredLeads = filteredLeads.filter(lead => lead.source === sourceFilter);
    }

    if (statusFilter) {
      filteredLeads = filteredLeads.filter(lead => lead.status === statusFilter);
    }

    setLeads(filteredLeads);
  };

  // Reset filters
  const resetFilters = () => {
    setSearchTerm('');
    setSourceFilter('');
    setStatusFilter('');
    setLeads(initialLeads);
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1>Leads Management</h1>
        <Link to="/add-lead" className="btn btn-primary">+ Add New Lead</Link>
      </div>

      {/* Search and Filter */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Search & Filter</h2>
          <button 
            className="btn btn-secondary btn-sm" 
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter size={16} className="mr-2" />
            {showFilters ? 'Hide Filters' : 'Show Filters'}
          </button>
        </div>

        <div className="flex items-center mb-4">
          <div className="relative w-full">
            <input
              type="text"
              placeholder="Search leads..."
              className="form-input pl-10"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          </div>
          <button className="btn btn-primary ml-4" onClick={applyFilters}>Search</button>
        </div>

        {showFilters && (
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Source</label>
              <select 
                className="form-select"
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
              >
                <option value="">All Sources</option>
                <option value="Website">Website</option>
                <option value="Email">Email</option>
                <option value="Social">Social</option>
                <option value="Referral">Referral</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Status</label>
              <select 
                className="form-select"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="">All Statuses</option>
                <option value="high">High Potential</option>
                <option value="medium">Medium Potential</option>
                <option value="low">Low Potential</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">&nbsp;</label>
              <button className="btn btn-secondary" onClick={resetFilters}>Reset Filters</button>
            </div>
          </div>
        )}
      </div>

      {/* Leads Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">All Leads</h2>
          <div>{leads.length} leads found</div>
        </div>
        
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th onClick={() => handleSort('name')} style={{ cursor: 'pointer' }}>
                  Name
                  {sortColumn === 'name' && (
                    sortDirection === 'asc' ? <ChevronUp size={16} className="inline ml-1" /> : <ChevronDown size={16} className="inline ml-1" />
                  )}
                </th>
                <th onClick={() => handleSort('email')} style={{ cursor: 'pointer' }}>
                  Email
                  {sortColumn === 'email' && (
                    sortDirection === 'asc' ? <ChevronUp size={16} className="inline ml-1" /> : <ChevronDown size={16} className="inline ml-1" />
                  )}
                </th>
                <th>Phone</th>
                <th onClick={() => handleSort('source')} style={{ cursor: 'pointer' }}>
                  Source
                  {sortColumn === 'source' && (
                    sortDirection === 'asc' ? <ChevronUp size={16} className="inline ml-1" /> : <ChevronDown size={16} className="inline ml-1" />
                  )}
                </th>
                <th onClick={() => handleSort('score')} style={{ cursor: 'pointer' }}>
                  Score
                  {sortColumn === 'score' && (
                    sortDirection === 'asc' ? <ChevronUp size={16} className="inline ml-1" /> : <ChevronDown size={16} className="inline ml-1" />
                  )}
                </th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {leads.map(lead => (
                <tr key={lead.id}>
                  <td>{lead.name}</td>
                  <td>{lead.email}</td>
                  <td>{lead.phone}</td>
                  <td>{lead.source}</td>
                  <td>{lead.score}/100</td>
                  <td>
                    <span className={`status-pill ${lead.status}`}>
                      {lead.status === 'high' ? 'High' : lead.status === 'medium' ? 'Medium' : 'Low'}
                    </span>
                  </td>
                  <td>
                    <button className="btn btn-secondary btn-sm mr-2">Edit</button>
                    <button className="btn btn-primary btn-sm">View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Leads;
