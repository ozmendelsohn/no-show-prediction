import { useMemo } from 'react';
import AppointmentCard from './AppointmentCard';
import { generateTimeSlots, generateWeekDays } from '@/utils/calendar';

export default function CalendarGrid() {
  const timeSlots = useMemo(() => generateTimeSlots(), []);
  const weekDays = useMemo(() => generateWeekDays(), []);

  return (
    <div className="grid grid-cols-8 border-r">
      {/* Time column */}
      <div className="col-span-1">
        <div className="h-20" /> {/* Empty header cell */}
        {timeSlots.map(time => (
          <div key={time} className="h-24 text-right pr-2 text-gray-500 text-xs">
            {time}
          </div>
        ))}
      </div>

      {/* Days columns */}
      {weekDays.map(day => (
        <div key={day.date} className="col-span-1">
          <div className="h-20 border-l border-b p-2 text-center">
            <div className="text-xs text-gray-500 uppercase">{day.name}</div>
            <div className="text-2xl mt-1">{day.number}</div>
          </div>
          {timeSlots.map(time => (
            <div key={time} className="h-24 border-l border-b p-1">
              {/* Appointments will be rendered here */}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
} 