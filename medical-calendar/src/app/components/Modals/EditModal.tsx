'use client';

import React from 'react';

export default function EditModal() {
  return (
    <div className="modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Edit Appointment</h2>
          <button className="close-button">&times;</button>
        </div>
        <div className="modal-body space-y-4">
          <div className="form-group">
            <label>Doctor</label>
            <input type="text" className="form-input" placeholder="Doctor's name" />
          </div>
          <div className="form-group">
            <label>Patient</label>
            <input type="text" className="form-input" placeholder="Patient's name" />
          </div>
          <div className="form-group">
            <label>Status</label>
            <select className="form-input">
              <option value="confirmed">Confirmed</option>
              <option value="pending">Pending</option>
            </select>
          </div>
          <div className="form-group">
            <label>Notes</label>
            <textarea className="form-input" rows={4} placeholder="Add appointment notes..." />
          </div>
        </div>
        <div className="modal-footer">
          <button className="modal-button secondary">Cancel</button>
          <button className="modal-button primary">Save Changes</button>
        </div>
      </div>
    </div>
  );
} 