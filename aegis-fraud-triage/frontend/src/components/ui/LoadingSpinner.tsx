import React from 'react';

interface LoadingSpinnerProps {
  label?: string;
}

export function LoadingSpinner({ label }: LoadingSpinnerProps) {
  return (
    <div className="text-gray-500 text-sm text-center py-8 flex flex-col items-center gap-2">
      <div className="w-5 h-5 border-2 border-white/30 border-t-transparent rounded-full animate-spin" />
      {label && <span>{label}</span>}
    </div>
  );
}
