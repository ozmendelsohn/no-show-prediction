import { Box, Button, Typography } from '@mui/material';
import { format, startOfWeek } from 'date-fns';
import CalendarGrid from './CalendarGrid';

interface CalendarProps {
  currentDate: Date;
}

const Calendar = ({ currentDate }: CalendarProps) => {
  const weekStart = startOfWeek(currentDate, { weekStartsOn: 0 });

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center',
          px: 3,
          py: 2,
          borderBottom: '1px solid #dadce0',
          bgcolor: '#fff',
          height: '64px',
          flexShrink: 0
        }}
      >
        <Typography 
          component="h1"
          sx={{ 
            fontSize: '22px',
            fontWeight: 400,
            color: '#3c4043',
            m: 0,
            pr: 3
          }}
        >
          Medical Appointments Calendar
        </Typography>
        <Button
          variant="contained"
          sx={{
            bgcolor: '#1a73e8',
            color: 'white',
            px: 3,
            py: 1,
            fontSize: '14px',
            '&:hover': {
              bgcolor: '#1557b0',
            },
          }}
        >
          Today
        </Button>
      </Box>

      {/* Calendar Grid */}
      <Box sx={{ 
        flex: 1, 
        overflow: 'auto', 
        position: 'relative',
        px: 2,
        pb: 2
      }}>
        <CalendarGrid weekStart={weekStart} currentDate={currentDate} />
      </Box>
    </Box>
  );
};

export default Calendar; 