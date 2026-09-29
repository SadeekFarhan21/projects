import { Loader2 } from 'lucide-react';
import { useStore } from '@/store/useStore';

export function ProcessingOverlay() {
  const { isProcessing, processingStatus } = useStore();

  if (!isProcessing || !processingStatus) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4 border border-gray-700">
        <div className="text-center">
          {/* Animated Icon */}
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-24 h-24 bg-primary-500/20 rounded-full animate-pulse-slow"></div>
            </div>
            <Loader2 className="w-16 h-16 text-primary-500 animate-spin mx-auto relative" />
          </div>

          {/* Status */}
          <h2 className="text-2xl font-bold text-white mb-2">
            {processingStatus.status === 'pending' && 'Initializing...'}
            {processingStatus.status === 'processing' && 'Processing PDFs'}
            {processingStatus.status === 'completed' && 'Complete!'}
            {processingStatus.status === 'failed' && 'Failed'}
          </h2>

          {/* Message */}
          <p className="text-gray-400 mb-6">{processingStatus.message}</p>

          {/* Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2.5 mb-4">
            <div
              className="bg-gradient-to-r from-primary-600 to-primary-500 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${processingStatus.progress * 100}%` }}
            ></div>
          </div>

          {/* Progress Percentage */}
          <p className="text-sm text-gray-500">
            {Math.round(processingStatus.progress * 100)}% Complete
          </p>
        </div>
      </div>
    </div>
  );
}

