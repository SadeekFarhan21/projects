import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from './components/ErrorBoundary';

const LandingPage = lazy(() => import('./landing/LandingPage'));
const DashboardApp = lazy(() => import('./DashboardApp'));

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <ErrorBoundary>
              <Suspense fallback={<div className="min-h-screen bg-black" />}>
                <LandingPage />
              </Suspense>
            </ErrorBoundary>
          }
        />
        <Route
          path="/demo"
          element={
            <ErrorBoundary>
              <Suspense
                fallback={
                  <div className="min-h-screen bg-[#050505] flex items-center justify-center text-white font-rajdhani">
                    Loading dashboard...
                  </div>
                }
              >
                <DashboardApp />
              </Suspense>
            </ErrorBoundary>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
