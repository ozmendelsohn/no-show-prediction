'use client';

import React, { createContext, useContext, useState } from 'react';
import { Appointment } from '../types';

interface ContextMenu {
  show: boolean;
  x: number;
  y: number;
}

interface AppointmentContextType {
  selectedAppointment: Appointment | null;
  setSelectedAppointment: (appointment: Appointment | null) => void;
  contextMenu: ContextMenu;
  setContextMenu: (menu: ContextMenu) => void;
}

const AppointmentContext = createContext<AppointmentContextType | undefined>(undefined);

function AppointmentProvider({ children }: { children: React.ReactNode }) {
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenu>({ show: false, x: 0, y: 0 });

  return (
    <AppointmentContext.Provider value={{
      selectedAppointment,
      setSelectedAppointment,
      contextMenu,
      setContextMenu
    }}>
      {children}
    </AppointmentContext.Provider>
  );
}

function useAppointmentContext() {
  const context = useContext(AppointmentContext);
  if (context === undefined) {
    throw new Error('useAppointmentContext must be used within an AppointmentProvider');
  }
  return context;
}

export { AppointmentProvider, useAppointmentContext }; 