import { Paper, Box, Typography, Grid } from '@mui/material';
import { startOfWeek, addDays, format } from 'date-fns';

interface WeekViewProps {
  currentDate: Date;
}

const HOURS = Array.from({ length: 24 }, (_, i) => i);
const DAYS = Array.from({ length: 7 }, (_, i) => i);

const WeekView = ({ currentDate }: WeekViewProps) => {
  const weekStart = startOfWeek(currentDate);

  return (
    <Paper sx={{ flex: 1, overflow: 'auto', m: 2 }}>
      {/* Header with days */}
      <Grid container sx={{ position: 'sticky', top: 0, bgcolor: 'background.paper', zIndex: 1 }}>
        <Grid item xs={1}>
          <Box sx={{ 
            height: 70, 
            borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
            borderRight: '1px solid rgba(0, 0, 0, 0.12)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }} />
        </Grid>
        {DAYS.map((dayOffset) => {
          const date = addDays(weekStart, dayOffset);
          return (
            <Grid item xs key={dayOffset}>
              <Box
                sx={{
                  height: 70,
                  borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
                  borderRight: '1px solid rgba(0, 0, 0, 0.12)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  bgcolor: 'background.paper',
                }}
              >
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                  {format(date, 'EEE')}
                </Typography>
                <Typography variant="h6" sx={{ color: 'text.secondary' }}>
                  {format(date, 'd')}
                </Typography>
              </Box>
            </Grid>
          );
        })}
      </Grid>

      {/* Time grid */}
      <Grid container>
        {/* Time labels */}
        <Grid item xs={1}>
          {HOURS.map((hour) => (
            <Box
              key={hour}
              sx={{
                height: 60,
                borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
                borderRight: '1px solid rgba(0, 0, 0, 0.12)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'background.paper',
              }}
            >
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {format(new Date().setHours(hour, 0), 'h a')}
              </Typography>
            </Box>
          ))}
        </Grid>

        {/* Time slots */}
        {DAYS.map((dayOffset) => (
          <Grid item xs key={dayOffset}>
            {HOURS.map((hour) => (
              <Box
                key={hour}
                sx={{
                  height: 60,
                  borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
                  borderRight: '1px solid rgba(0, 0, 0, 0.12)',
                  '&:hover': {
                    backgroundColor: 'action.hover',
                    cursor: 'pointer',
                  },
                }}
              />
            ))}
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default WeekView; 
 