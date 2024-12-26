import { AppointmentProvider } from '@/contexts/AppointmentContext';
import CalendarHeader from '@/components/Calendar/CalendarHeader';
import CalendarGrid from '@/components/Calendar/CalendarGrid';
import ContextMenu from '@/components/Modals/ContextMenu';
import EditModal from '@/components/Modals/EditModal';
import RescheduleModal from '@/components/Modals/RescheduleModal';
import CallModal from '@/components/Modals/CallModal';
import SplitModal from '@/components/Modals/SplitModal';
import Toast from '@/components/Toast';

export default function Home() {
  return (
    <AppointmentProvider>
      <div className="min-h-screen bg-white">
        <main className="max-w-[1400px] mx-auto p-4">
          <CalendarHeader />
          <div className="mt-6">
            <CalendarGrid />
          </div>
        </main>

        <ContextMenu />
        <EditModal />
        <RescheduleModal />
        <CallModal />
        <SplitModal />
        <Toast />
      </div>
    </AppointmentProvider>
  );
} 