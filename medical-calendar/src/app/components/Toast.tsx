'use client';

export default function Toast() {
  return (
    <div className="fixed bottom-6 right-6 bg-gray-800 text-white px-6 py-3 rounded-lg hidden">
      <span className="material-icons mr-2">check_circle</span>
      <span className="message"></span>
    </div>
  );
} 