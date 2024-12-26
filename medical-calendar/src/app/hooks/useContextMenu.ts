'use client';

import { useState, useEffect } from 'react';

interface ContextMenuPosition {
  x: number;
  y: number;
}

interface Appointment {
  id: string;
  doctorName: string;
  time: string;
  type: 'general' | 'yellow' | 'cardiology';
  specialty?: string;
  day: number;
}

export function useContextMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<ContextMenuPosition>({ x: 0, y: 0 });
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const menu = document.getElementById('context-menu');
      if (menu && !menu.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const openMenu = (event: React.MouseEvent, appointment: Appointment) => {
    event.stopPropagation();
    
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const menuWidth = 200;
    const menuHeight = 300;
    
    let x = rect.right + 10;
    let y = rect.top;

    // Adjust position if menu would go off screen
    if (x + menuWidth > window.innerWidth) {
      x = rect.left - menuWidth - 10;
    }
    if (y + menuHeight > window.innerHeight) {
      y = window.innerHeight - menuHeight - 10;
    }

    setPosition({ x, y });
    setSelectedAppointment(appointment);
    setIsOpen(true);
  };

  return {
    isOpen,
    position,
    selectedAppointment,
    openMenu,
    setIsOpen,
  };
} 