'use client';

import React from 'react';

export default function CallModal() {
  return (
    <div className="modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Make a Call</h2>
          <button className="close-button">&times;</button>
        </div>
        <div className="modal-body space-y-4">
          <div className="form-group">
            <label>Call Type</label>
            <select className="form-input">
              <option value="reminder">Appointment Reminder</option>
              <option value="confirmation">Appointment Confirmation</option>
              <option value="cancellation">Cancellation Notice</option>
              <option value="custom">Other</option>
            </select>
          </div>
          <div className="form-group">
            <label>Notes for Call</label>
            <textarea 
              className="form-input" 
              rows={4} 
              placeholder="Add notes about the call here..."
            />
          </div>
        </div>
        <div className="modal-footer">
          <button className="modal-button secondary">Cancel</button>
          <button className="modal-button primary">
            <span className="material-icons text-base">phone</span>
            Make Call
          </button>
        </div>
      </div>
    </div>
  );
} 