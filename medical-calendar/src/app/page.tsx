'use client';

import React, { useState } from 'react';
import { AppointmentProvider } from './contexts/AppointmentContext';
import CalendarHeader from './components/Calendar/CalendarHeader';
import CalendarGrid from './components/Calendar/CalendarGrid';
import ContextMenu from './components/Modals/ContextMenu';
import EditModal from './components/Modals/EditModal';
import RescheduleModal from './components/Modals/RescheduleModal';
import CallModal from './components/Modals/CallModal';
import SplitModal from './components/Modals/SplitModal';
import Toast from './components/Toast';

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <AppointmentProvider>
      <div className="flex h-screen bg-gray-50">
        {/* Sidebar */}
        <div className={`${sidebarCollapsed ? 'w-20' : 'w-64'} bg-white shadow-lg border-r border-gray-200 transition-all duration-300 ease-in-out flex flex-col`}>
          <div className="p-6 border-b border-gray-100">
            <div className={`flex items-center ${sidebarCollapsed ? 'justify-center' : 'justify-between'}`}>
              {!sidebarCollapsed && (
                <div>
                  <h1 className="text-xl font-bold text-gray-800">KeepTime Health</h1>
                  <p className="text-xs text-gray-500 mt-1">Smart Healthcare Suite</p>
                </div>
              )}
              <button 
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <span className="material-icons text-gray-400">
                  {sidebarCollapsed ? 'menu_open' : 'menu'}
                </span>
              </button>
            </div>
          </div>
          
          <div className="flex-1 p-4">
            <div className="space-y-2">
              {['dashboard', 'calendar_today', 'people', 'medical_services', 'settings', 'help'].map((icon, index) => (
                <button 
                  key={icon}
                  className={`w-full text-left rounded-xl p-3 hover:bg-gray-50 transition-all duration-200 flex items-center gap-3
                    ${index === 1 ? 'bg-indigo-50 text-indigo-600 shadow-sm' : 'text-gray-700'}`}
                >
                  <span className={`material-icons ${index === 1 ? 'text-indigo-600' : 'text-gray-400'}`}>
                    {icon}
                  </span>
                  {!sidebarCollapsed && (
                    <span className="font-medium">
                      {icon.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="p-4 border-t border-gray-100">
            {!sidebarCollapsed ? (
              <div className="flex items-center gap-3 p-3 rounded-xl bg-gray-50">
                <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                  <span className="text-sm font-bold text-indigo-600">DR</span>
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium text-gray-800">Dr. Roberts</div>
                  <div className="text-xs text-gray-500">Cardiologist</div>
                </div>
              </div>
            ) : (
              <div className="flex justify-center">
                <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                  <span className="text-sm font-bold text-indigo-600">DR</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Top Navigation */}
          <div className="h-16 bg-white shadow-sm flex items-center justify-between px-6">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 text-gray-500">
                <span className="material-icons">event</span>
                <span className="text-sm font-medium">March 2024</span>
              </div>
              <div className="h-6 w-px bg-gray-200"></div>
              <div className="flex gap-2">
                <button className="px-3 py-1.5 text-sm font-medium text-indigo-600 bg-indigo-50 rounded-lg hover:bg-indigo-100 transition-colors">
                  + New Appointment
                </button>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button className="p-2 hover:bg-gray-50 rounded-lg relative group">
                <span className="material-icons text-gray-400">notifications</span>
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                <span className="absolute top-full right-0 mt-2 w-48 bg-white shadow-lg rounded-lg py-2 px-3 text-sm text-gray-700 opacity-0 group-hover:opacity-100 transition-opacity">
                  3 new notifications
                </span>
              </button>
              <div className="h-6 w-px bg-gray-200"></div>
              <button className="flex items-center gap-2 px-3 py-1.5 hover:bg-gray-50 rounded-lg group relative">
                <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-medium">
                  DR
                </div>
                <span className="text-sm text-gray-700">Dr. Roberts</span>
                <span className="material-icons text-gray-400 text-sm">arrow_drop_down</span>
              </button>
            </div>
          </div>

          {/* Calendar Content */}
          <div className="flex-1 overflow-auto p-6 bg-gradient-to-br from-gray-50 to-white">
            <div className="max-w-[1400px] mx-auto space-y-6">
              <CalendarHeader />
              <CalendarGrid />
            </div>
          </div>
        </div>
      </div>

      {/* Modals */}
      <ContextMenu />
      <EditModal />
      <RescheduleModal />
      <CallModal />
      <SplitModal />
      <Toast />
    </AppointmentProvider>
  );
} 