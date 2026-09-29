import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-4 text-center">
          <div className="text-red-400 text-sm font-semibold">Component Error</div>
          <div className="text-gray-500 text-xs mt-1">{this.state.error?.message}</div>
        </div>
      );
    }
    return this.props.children;
  }
}
