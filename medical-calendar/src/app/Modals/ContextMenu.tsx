import { useAppointmentContext } from '@/contexts/AppointmentContext';

export default function ContextMenu() {
  const { contextMenu, setContextMenu, selectedAppointment } = useAppointmentContext();

  if (!contextMenu.show || !selectedAppointment) return null;

  const menuItems = [
    { icon: 'edit', label: 'Edit' },
    { icon: 'schedule', label: 'Reschedule' },
    { icon: 'phone', label: 'Give a Call' },
    { icon: 'call_split', label: 'Split Appointment' }
  ];

  return (
    <div
      className="fixed bg-white shadow-lg rounded-lg py-2 min-w-[200px] z-50"
      style={{ top: contextMenu.y, left: contextMenu.x }}
    >
      {menuItems.map((item, index) => (
        <>
          <div
            key={item.label}
            className="px-4 py-2 hover:bg-gray-50 cursor-pointer flex items-center"
          >
            <span className="material-icons mr-3 text-gray-500">{item.icon}</span>
            {item.label}
          </div>
          {index === 1 && <hr className="my-2" />}
        </>
      ))}
    </div>
  );
} 