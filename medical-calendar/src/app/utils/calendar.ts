export function generateTimeSlots() {
  const slots = [];
  for (let i = 8; i <= 17; i++) {
    slots.push(`${i.toString().padStart(2, '0')}:00`);
  }
  return slots;
}

export function generateWeekDays() {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const startDate = new Date(2024, 2, 18); // March 18, 2024

  return days.map((name, index) => {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + index);
    return {
      name,
      number: date.getDate(),
      date: date.toISOString().split('T')[0],
    };
  });
} 