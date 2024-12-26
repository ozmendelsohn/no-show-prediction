'use client';

import React from 'react';

export default function SplitModal() {
  return (
    <div className="modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Split Appointment</h2>
          <button className="close-button">&times;</button>
        </div>
        <div className="modal-body space-y-4">
          <div className="form-group">
            <label>Select Patient by Acuity Score</label>
            <div className="patient-list">
              {/* Patient list will be populated dynamically */}
            </div>
          </div>
          <div className="space-y-3">
            <label className="text-sm font-medium text-gray-700">Time Slots</label>
            <div id="splitTimeSlots" className="space-y-2">
              {/* Time slots will be shown after patient selection */}
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button className="modal-button secondary">Cancel</button>
          <button className="modal-button primary">Assign Selected Patient</button>
        </div>
      </div>
    </div>
  );
} 