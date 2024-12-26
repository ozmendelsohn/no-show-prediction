'use client';

import React from 'react';
import { Appointment } from '@/app/types';
import { useAppointmentContext } from '@/app/contexts/AppointmentContext';

interface AppointmentCardProps {
  appointment: Appointment;
}

const typeStyles = {
  general: 'bg-blue-50 text-blue-700 border-blue-200',
  cardiology: 'bg-red-50 text-red-700 border-red-200',
  dentistry: 'bg-green-50 text-green-700 border-green-200',
  orthopedics: 'bg-orange-50 text-orange-700 border-orange-200',
  yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200'
};

export default function AppointmentCard({ appointment }: AppointmentCardProps) {
  const { setSelectedAppointment, setContextMenu } = useAppointmentContext();

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setSelectedAppointment(appointment);
    setContextMenu({
      show: true,
      x: e.clientX,
      y: e.clientY
    });
  };

  return (
    <div
      className={`p-3 rounded-lg cursor-pointer ${typeStyles[appointment.type]} border shadow-sm hover:shadow-md transition-all duration-200`}
      onClick={handleClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-sm font-medium">
            {appointment.doctorName}
          </span>
          <span className="text-xs opacity-75">
            {appointment.type}
          </span>
        </div>
        {appointment.status === 'confirmed' && (
          <div className="w-2 h-2 rounded-full bg-green-500" />
        )}
      </div>
      <div className="text-xs mt-2 opacity-75 flex items-center">
        <span className="material-icons text-sm mr-1">schedule</span>
        {appointment.time}
      </div>
    </div>
  );
} 