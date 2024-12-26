'use client';

export default function RescheduleModal() {
  return (
    <div className="modal">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Reschedule Appointment</h2>
          <button className="close-button">&times;</button>
        </div>
        <div className="modal-body">
          <div className="form-group">
            <label>Date</label>
            <input type="date" className="form-input" />
          </div>
          <div className="form-group">
            <label>Time</label>
            <select className="form-input">
              {Array.from({ length: 10 }, (_, i) => i + 8).map(hour => (
                <option key={hour} value={hour}>
                  {`${hour.toString().padStart(2, '0')}:00`}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="modal-footer">
          <button className="modal-button secondary">Cancel</button>
          <button className="modal-button primary">Reschedule</button>
        </div>
      </div>
    </div>
  );
} 