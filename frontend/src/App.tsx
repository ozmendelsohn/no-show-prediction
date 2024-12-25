import { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import Calendar from './components/Calendar';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1a73e8',
    },
    text: {
      primary: '#3c4043',
      secondary: '#70757a',
    },
    background: {
      default: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  const [currentDate] = useState(new Date());

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh',
        width: '100%',
        bgcolor: 'background.default',
        p: 3
      }}>
        <Box sx={{
          maxWidth: '1400px',
          margin: '0 auto',
          height: 'calc(100vh - 48px)',
          bgcolor: '#fff',
          borderRadius: 1,
          boxShadow: '0 1px 3px 0 rgb(60 64 67 / 30%)',
          overflow: 'hidden'
        }}>
          <Calendar currentDate={currentDate} />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
