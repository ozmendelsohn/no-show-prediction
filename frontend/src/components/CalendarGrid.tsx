import { Box, Typography } from '@mui/material';
import { format, addDays, isSameDay } from 'date-fns';

interface CalendarGridProps {
  weekStart: Date;
  currentDate: Date;
}

interface Appointment {
  id: string;
  title: string;
  time: string;
  type: 'general' | 'cardiology' | 'dentistry' | 'orthopedics';
}

const HOURS = Array.from({ length: 10 }, (_, i) => i + 8); // 8 AM to 5 PM
const DAYS = Array.from({ length: 7 }, (_, i) => i);

const appointmentColors = {
  general: {
    bg: '#d2e3fc',
    text: '#1a73e8'
  },
  cardiology: {
    bg: '#fad1d1',
    text: '#c53929'
  },
  dentistry: {
    bg: '#ceead6',
    text: '#137333'
  },
  orthopedics: {
    bg: '#feefc3',
    text: '#b06000'
  }
};

const CalendarGrid = ({ weekStart, currentDate }: CalendarGridProps) => {
  return (
    <Box sx={{ 
      display: 'grid',
      gridTemplateColumns: '60px repeat(7, 1fr)',
      borderRight: '1px solid #dadce0',
      height: '100%',
      overflow: 'auto',
      bgcolor: '#fff',
      borderRadius: 1
    }}>
      {/* Empty corner cell */}
      <Box sx={{ 
        bgcolor: '#fff', 
        borderBottom: '1px solid #dadce0',
        position: 'sticky',
        top: 0,
        zIndex: 2
      }}/>
      
      {/* Days header */}
      {DAYS.map((dayOffset) => {
        const date = addDays(weekStart, dayOffset);
        const isToday = isSameDay(date, new Date());
        
        return (
          <Box
            key={dayOffset}
            sx={{
              py: 1,
              px: 0.5,
              textAlign: 'center',
              borderLeft: '1px solid #dadce0',
              borderBottom: '1px solid #dadce0',
              bgcolor: '#fff',
              position: 'sticky',
              top: 0,
              zIndex: 2
            }}
          >
            <Typography
              sx={{
                fontSize: '11px',
                color: '#70757a',
                textTransform: 'uppercase',
              }}
            >
              {format(date, 'EEE')}
            </Typography>
            <Typography
              sx={{
                fontSize: '24px',
                color: isToday ? '#1a73e8' : '#70757a',
                fontWeight: isToday ? 500 : 400,
                mt: 0.5,
              }}
            >
              {format(date, 'd')}
            </Typography>
          </Box>
        );
      })}

      {/* Time grid */}
      {HOURS.map((hour) => (
        <>
          {/* Time label */}
          <Box
            key={`time-${hour}`}
            sx={{
              color: '#70757a',
              fontSize: '10px',
              pr: 2,
              textAlign: 'right',
              borderBottom: '1px solid #dadce0',
              height: '48px',
              display: 'flex',
              alignItems: 'start',
              pt: 1,
              bgcolor: '#fff',
              position: 'sticky',
              left: 0,
              zIndex: 1
            }}
          >
            {format(new Date().setHours(hour, 0), 'h:mm a')}
          </Box>

          {/* Time slots */}
          {DAYS.map((dayOffset) => (
            <Box
              key={`slot-${hour}-${dayOffset}`}
              sx={{
                borderLeft: '1px solid #dadce0',
                borderBottom: '1px solid #dadce0',
                height: '48px',
                '&:hover': {
                  bgcolor: '#f8f9fa',
                },
                position: 'relative',
              }}
            />
          ))}
        </>
      ))}
    </Box>
  );
};

export default CalendarGrid; 