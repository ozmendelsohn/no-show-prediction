'use client';

import React from 'react';
import { useContextMenu } from '../../hooks/useContextMenu';

const timeSlots = Array.from({ length: 9 }, (_, i) => i + 9); // 9 AM to 5 PM
const weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];

interface Appointment {
  id: string;
  doctorName: string;
  time: string;
  type: 'general' | 'yellow' | 'cardiology';
  specialty?: string;
  day: number; // 0-4 for Monday-Friday
}

const appointments: Appointment[] = [
  // Monday - General (Blue) Appointments
  { id: '1', doctorName: 'Dr. Johnson', time: '9:00', type: 'general', specialty: 'General Practice', day: 0 },
  { id: '2', doctorName: 'Dr. Lee', time: '10:30', type: 'general', specialty: 'Family Medicine', day: 0 },
  { id: '3', doctorName: 'Dr. Patel', time: '13:00', type: 'general', specialty: 'Internal Medicine', day: 0 },
  { id: '4', doctorName: 'Dr. Davis', time: '14:30', type: 'general', specialty: 'General Practice', day: 0 },
  { id: '5', doctorName: 'Dr. Kim', time: '16:00', type: 'general', specialty: 'Family Medicine', day: 0 },

  // Tuesday - Mix of Appointments
  { id: '6', doctorName: 'Dr. White', time: '9:30', type: 'yellow', specialty: 'Pediatrics', day: 1 },
  { id: '7', doctorName: 'Dr. Miller', time: '11:00', type: 'yellow', specialty: 'Orthopedics', day: 1 },
  { id: '8', doctorName: 'Dr. Roberts', time: '13:30', type: 'cardiology', specialty: 'Cardiology', day: 1 },
  { id: '9', doctorName: 'Dr. Zhang', time: '15:00', type: 'general', specialty: 'Internal Medicine', day: 1 },
  { id: '10', doctorName: 'Dr. Brown', time: '16:30', type: 'yellow', specialty: 'ENT', day: 1 },

  // Wednesday - More Appointments
  { id: '11', doctorName: 'Dr. Thompson', time: '9:00', type: 'yellow', specialty: 'Neurology', day: 2 },
  { id: '12', doctorName: 'Dr. Chen', time: '10:30', type: 'cardiology', specialty: 'Cardiology', day: 2 },
  { id: '13', doctorName: 'Dr. Evans', time: '13:00', type: 'yellow', specialty: 'Psychiatry', day: 2 },
  { id: '14', doctorName: 'Dr. Wilson', time: '14:30', type: 'cardiology', specialty: 'Cardiology', day: 2 },
  { id: '15', doctorName: 'Dr. Kumar', time: '16:00', type: 'general', specialty: 'Family Medicine', day: 2 },

  // Thursday - Mixed Schedule
  { id: '16', doctorName: 'Dr. Bailey', time: '9:00', type: 'yellow', specialty: 'Ophthalmology', day: 3 },
  { id: '17', doctorName: 'Dr. Cohen', time: '10:30', type: 'general', specialty: 'General Practice', day: 3 },
  { id: '18', doctorName: 'Dr. Novak', time: '13:00', type: 'cardiology', specialty: 'Cardiology', day: 3 },
  { id: '19', doctorName: 'Dr. Singh', time: '14:30', type: 'general', specialty: 'Internal Medicine', day: 3 },
  { id: '20', doctorName: 'Dr. Lopez', time: '16:00', type: 'yellow', specialty: 'Dermatology', day: 3 },

  // Friday - Full Schedule
  { id: '21', doctorName: 'Dr. Fischer', time: '9:00', type: 'general', specialty: 'Family Medicine', day: 4 },
  { id: '22', doctorName: 'Dr. Ivanov', time: '10:30', type: 'yellow', specialty: 'Orthopedics', day: 4 },
  { id: '23', doctorName: 'Dr. Ahmed', time: '13:00', type: 'cardiology', specialty: 'Cardiology', day: 4 },
  { id: '24', doctorName: 'Dr. Torres', time: '14:30', type: 'general', specialty: 'General Practice', day: 4 },
  { id: '25', doctorName: 'Dr. Gray', time: '16:00', type: 'yellow', specialty: 'Neurology', day: 4 },

  // Additional Appointments Throughout the Week
  { id: '26', doctorName: 'Dr. Müller', time: '11:00', type: 'general', specialty: 'Internal Medicine', day: 0 },
  { id: '27', doctorName: 'Dr. Price', time: '12:30', type: 'yellow', specialty: 'Pediatrics', day: 1 },
  { id: '28', doctorName: 'Dr. Sanders', time: '15:30', type: 'cardiology', specialty: 'Cardiology', day: 2 },
  { id: '29', doctorName: 'Dr. Wong', time: '12:00', type: 'general', specialty: 'Family Medicine', day: 3 },
  { id: '30', doctorName: 'Dr. Kim', time: '11:30', type: 'yellow', specialty: 'Dermatology', day: 4 }
];

const getAppointmentStyle = (type: string) => {
  switch (type) {
    case 'yellow':
      return 'bg-[#fff8c5] text-[#806102]';
    case 'cardiology':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-blue-100 text-blue-800';
  }
};

export default function CalendarGrid() {
  const { isOpen, position, selectedAppointment, openMenu, setIsOpen } = useContextMenu();

  return (
    <div className="relative" onClick={() => setIsOpen(false)}>
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
        {/* Grid Header */}
        <div className="grid grid-cols-6 border-b">
          <div className="p-4 font-medium text-gray-400 text-sm">Time</div>
          {weekDays.map((day) => (
            <div key={day} className="p-4 font-medium text-gray-800">
              {day}
            </div>
          ))}
        </div>

        {/* Grid Body */}
        <div className="divide-y">
          {timeSlots.map((hour) => (
            <div key={hour} className="grid grid-cols-6">
              {/* Time Column */}
              <div className="p-4 text-sm text-gray-400 border-r">
                {`${hour}:00`}
              </div>

              {/* Appointment Slots */}
              {weekDays.map((day, dayIndex) => {
                const dayAppointments = appointments.filter(apt => {
                  const aptHour = parseInt(apt.time.split(':')[0]);
                  return aptHour === hour && apt.day === dayIndex;
                });

                return (
                  <div key={`${day}-${hour}`} className="p-2 border-r min-h-[100px] relative">
                    {dayAppointments.map((apt) => (
                      <div
                        key={apt.id}
                        onClick={(e) => openMenu(e, apt)}
                        className={`mb-2 p-3 rounded-xl cursor-pointer transition-all duration-200 hover:shadow-lg ${getAppointmentStyle(apt.type)}`}
                      >
                        <div className="font-medium">{apt.doctorName}</div>
                        <div className="text-sm opacity-75">{apt.specialty}</div>
                        <div className="text-xs mt-1">{apt.time}</div>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Context Menu */}
      {isOpen && (
        <div
          id="context-menu"
          className="fixed z-50 min-w-[200px] bg-white rounded-lg shadow-xl border border-gray-100 overflow-hidden animate-fadeIn"
          style={{
            top: `${position.y}px`,
            left: `${position.x}px`,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="py-2">
            {/* Appointment Info Header */}
            <div className="px-4 py-2 bg-gray-50 border-b border-gray-100">
              <div className="font-medium text-gray-800">{selectedAppointment?.doctorName}</div>
              <div className="text-sm text-gray-500">{selectedAppointment?.specialty}</div>
              <div className="text-xs text-gray-400 mt-1">{selectedAppointment?.time}</div>
            </div>

            <button
              className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 transition-colors"
              onClick={() => {
                setIsOpen(false);
                // Handle edit
                console.log('Edit', selectedAppointment);
              }}
            >
              <span className="material-icons text-gray-400">edit</span>
              <span>Edit Appointment</span>
            </button>
            
            <button
              className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 transition-colors"
              onClick={() => {
                setIsOpen(false);
                // Handle reschedule
                console.log('Reschedule', selectedAppointment);
              }}
            >
              <span className="material-icons text-gray-400">event</span>
              <span>Reschedule</span>
            </button>
            
            <button
              className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 transition-colors"
              onClick={() => {
                setIsOpen(false);
                // Handle call
                console.log('Call', selectedAppointment);
              }}
            >
              <span className="material-icons text-gray-400">phone</span>
              <span>Give a Call</span>
            </button>
            
            <button
              className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 transition-colors"
              onClick={() => {
                setIsOpen(false);
                // Handle split
                console.log('Split', selectedAppointment);
              }}
            >
              <span className="material-icons text-gray-400">call_split</span>
              <span>Split Appointment</span>
            </button>

            <div className="border-t my-2"></div>
            
            <button
              className="w-full px-4 py-2 text-left hover:bg-gray-50 flex items-center gap-3 text-red-600 transition-colors"
              onClick={() => {
                setIsOpen(false);
                // Handle cancel
                console.log('Cancel', selectedAppointment);
              }}
            >
              <span className="material-icons text-red-400">cancel</span>
              <span>Cancel Appointment</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 