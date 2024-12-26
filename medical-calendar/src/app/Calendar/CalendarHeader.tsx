import { useState } from 'react';

export default function CalendarHeader() {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-normal">Medical Appointments Calendar</h1>
        <button className="bg-blue-600 text-white px-6 py-2 rounded">
          Today
        </button>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="flex gap-2">
          <button className="p-2 border rounded hover:bg-gray-50">
            <span className="material-icons">chevron_left</span>
          </button>
          <button className="p-2 border rounded hover:bg-gray-50">
            <span className="material-icons">chevron_right</span>
          </button>
        </div>
        
        <div className="text-xl">March 18 - 24, 2024</div>
        
        <button className="bg-blue-600 text-white px-6 py-2 rounded">
          Today
        </button>
        
        <div className="flex border rounded overflow-hidden">
          <button className="px-4 py-2 hover:bg-gray-50">Day</button>
          <button className="px-4 py-2 bg-blue-600 text-white">Week</button>
          <button className="px-4 py-2 hover:bg-gray-50">Month</button>
        </div>
      </div>
    </div>
  );
} 