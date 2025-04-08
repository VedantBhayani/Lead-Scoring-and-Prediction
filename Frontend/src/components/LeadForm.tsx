
import React, { useState } from 'react';

const LeadForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    company: '',
    source: '',
    notes: ''
  });

  const [errors, setErrors] = useState({
    name: '',
    email: '',
    phone: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });

    // Clear error when user types
    if (errors[name as keyof typeof errors]) {
      setErrors({
        ...errors,
        [name]: ''
      });
    }
  };

  const validateForm = () => {
    let valid = true;
    const newErrors = { name: '', email: '', phone: '' };

    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
      valid = false;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
      valid = false;
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid email';
      valid = false;
    }

    // Phone validation (simple validation for demo)
    const phoneRegex = /^\(\d{3}\) \d{3}-\d{4}$/;
    if (formData.phone && !phoneRegex.test(formData.phone)) {
      newErrors.phone = 'Please use format: (555) 555-5555';
      valid = false;
    }

    setErrors(newErrors);
    return valid;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      // Form is valid, submit data
      console.log('Form submitted:', formData);
      // Here you would typically send the data to your backend
      alert('Lead successfully added!');
      
      // Reset form
      setFormData({
        name: '',
        email: '',
        phone: '',
        company: '',
        source: '',
        notes: ''
      });
    }
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1>Add New Lead</h1>
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Lead Information</h2>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="name" className="form-label">Full Name*</label>
              <input
                type="text"
                id="name"
                name="name"
                className={`form-input ${errors.name ? 'border-red-500' : ''}`}
                value={formData.name}
                onChange={handleChange}
              />
              {errors.name && <div className="text-red-500 text-sm mt-1">{errors.name}</div>}
            </div>

            <div className="form-group">
              <label htmlFor="email" className="form-label">Email Address*</label>
              <input
                type="email"
                id="email"
                name="email"
                className={`form-input ${errors.email ? 'border-red-500' : ''}`}
                value={formData.email}
                onChange={handleChange}
              />
              {errors.email && <div className="text-red-500 text-sm mt-1">{errors.email}</div>}
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="phone" className="form-label">Phone Number</label>
              <input
                type="tel"
                id="phone"
                name="phone"
                placeholder="(555) 555-5555"
                className={`form-input ${errors.phone ? 'border-red-500' : ''}`}
                value={formData.phone}
                onChange={handleChange}
              />
              {errors.phone && <div className="text-red-500 text-sm mt-1">{errors.phone}</div>}
            </div>

            <div className="form-group">
              <label htmlFor="company" className="form-label">Company</label>
              <input
                type="text"
                id="company"
                name="company"
                className="form-input"
                value={formData.company}
                onChange={handleChange}
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="source" className="form-label">Lead Source*</label>
            <select
              id="source"
              name="source"
              className="form-select"
              value={formData.source}
              onChange={handleChange}
              required
            >
              <option value="">Select a source</option>
              <option value="Website">Website</option>
              <option value="Email">Email Campaign</option>
              <option value="Social">Social Media</option>
              <option value="Referral">Referral</option>
              <option value="Event">Event</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="notes" className="form-label">Additional Notes</label>
            <textarea
              id="notes"
              name="notes"
              className="form-input"
              rows={4}
              value={formData.notes}
              onChange={handleChange}
            ></textarea>
          </div>

          <div className="mt-4">
            <button type="submit" className="btn btn-primary">
              Add Lead
            </button>
            <button type="button" className="btn btn-secondary ml-2">
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LeadForm;
