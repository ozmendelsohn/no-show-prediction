import { Appointment } from '@/types';
import { useAppointmentContext } from '@/contexts/AppointmentContext';

interface AppointmentCardProps {
  appointment: Appointment;
}

const typeStyles = {
  general: 'bg-blue-50 text-blue-700',
  cardiology: 'bg-red-50 text-red-700',
  dentistry: 'bg-green-50 text-green-700',
  orthopedics: 'bg-orange-50 text-orange-700',
  yellow: 'bg-yellow-50 text-yellow-700'
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
      className={`p-2 rounded cursor-pointer ${typeStyles[appointment.type]} hover:shadow-md transition-shadow`}
      onClick={handleClick}
    >
      <div className="text-sm font-medium">
        {appointment.doctorName} - {appointment.type}
      </div>
      <div className="text-xs opacity-70 mt-1">
        {appointment.time}
      </div>
      {appointment.status === 'confirmed' && (
        <div className="w-2 h-2 rounded-full bg-green-500 absolute right-2 top-1/2 transform -translate-y-1/2" />
      )}
    </div>
  );
} 