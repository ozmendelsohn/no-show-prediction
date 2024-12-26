export interface Appointment {
  id: string;
  doctorName: string;
  patientName?: string;
  type: 'general' | 'cardiology' | 'dentistry' | 'orthopedics' | 'yellow';
  time: string;
  date: string;
  status: 'confirmed' | 'pending';
}

export interface Patient {
  id: string;
  name: string;
  acuityScore: number;
} 