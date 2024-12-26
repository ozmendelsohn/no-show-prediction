'use client';

import React from 'react';

export default function CalendarHeader() {
  return (
    <div className="flex flex-col gap-6 bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold text-gray-800">Medical Appointments Calendar</h1>
        <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg transition-colors duration-200 font-medium">
          Today
        </button>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="flex gap-2">
          <button className="p-2 border rounded-lg hover:bg-gray-50 transition-colors duration-200 text-gray-600">
            <span className="material-icons">chevron_left</span>
          </button>
          <button className="p-2 border rounded-lg hover:bg-gray-50 transition-colors duration-200 text-gray-600">
            <span className="material-icons">chevron_right</span>
          </button>
        </div>
        
        <div className="text-xl font-medium text-gray-700">March 18 - 24, 2024</div>
        
        <div className="flex border rounded-lg overflow-hidden shadow-sm">
          <button className="px-6 py-2 hover:bg-gray-50 transition-colors duration-200 text-gray-700">Day</button>
          <button className="px-6 py-2 bg-indigo-600 text-white font-medium">Week</button>
          <button className="px-6 py-2 hover:bg-gray-50 transition-colors duration-200 text-gray-700">Month</button>
        </div>
      </div>
    </div>
  );
} 