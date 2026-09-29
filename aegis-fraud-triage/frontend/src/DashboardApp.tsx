import React, { useState, useEffect, useMemo } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { BentoGrid } from './components/layout/BentoGrid';
import { StatsOverview } from './components/tiles/StatsOverview';
import { TransactionFeed } from './components/tiles/TransactionFeed';
import { AlertPanel } from './components/tiles/AlertPanel';
import { NetworkGraph } from './components/tiles/NetworkGraph';
import { AgentConsole } from './components/tiles/AgentConsole';
import { DisputePanel } from './components/tiles/DisputePanel';
import { AccountActivity } from './components/tiles/AccountActivity';
import { InfraStatus } from './components/tiles/InfraStatus';
import { Shield, Wifi, WifiOff, Database, Clock, Maximize, Minimize, Server, X, Zap } from 'lucide-react';
import { API_URL } from './config';
import './App.css';

function DashboardApp() {
  const { transactions, alerts, agentTraces, disputes, stats, isConnected } = useWebSocket();
  const [nessieConnected, setNessieConnected] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showStatus, setShowStatus] = useState(false);
  const [serviceCount, setServiceCount] = useState({ online: 0, total: 0 });
  const [demoRunning, setDemoRunning] = useState(false);

  // Derive active investigation IDs from agent traces (supports parallel agents)
  const activeInvestigationIds = useMemo(() => {
    const groups: Record<string, typeof agentTraces> = {};
    for (const trace of agentTraces) {
      if (!groups[trace.alert_id]) groups[trace.alert_id] = [];
      groups[trace.alert_id].push(trace);
    }
    const active = new Set<string>();
    for (const alertId of Object.keys(groups)) {
      if (!groups[alertId].some(t => t.step_type === 'verdict')) {
        active.add(alertId);
      }
    }
    return active;
  }, [agentTraces]);

  // Sync fullscreen state with browser
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  const toggleDemo = async () => {
    try {
      const endpoint = demoRunning ? 'stop' : 'start';
      const res = await fetch(`${API_URL}/api/demo/${endpoint}`, { method: 'POST' });
      const data = await res.json();
      setDemoRunning(endpoint === 'start' && (data.status === 'started' || data.status === 'already_running'));
    } catch {}
  };

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check backend health + demo status
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/api/health`);
        const data = await res.json();
        setNessieConnected(data.nessie_connected || false);
        // Count online services for the status badge
        const services = [
          data.nessie_connected,
          data.anomaly_engine,
          data.vector_store_enabled,
          data.neo4j_connected,
          data.redis_connected,
          data.producer_active,
        ];
        setServiceCount({
          online: services.filter(Boolean).length,
          total: services.length,
        });
      } catch {
        setNessieConnected(false);
        setServiceCount({ online: 0, total: 7 });
      }
      // Poll demo status
      try {
        const demoRes = await fetch(`${API_URL}/api/demo/status`);
        const demoData = await demoRes.json();
        setDemoRunning(demoData.running);
      } catch {}
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Close modal on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowStatus(false);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  const allOnline = serviceCount.online === serviceCount.total && serviceCount.total > 0;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-hidden">
      {/* Animated background gradient */}
      <div className="fixed inset-0 bg-gradient-mesh pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 h-16 flex items-center justify-between px-6 border-b border-gray-800/40 bg-gray-950/60 backdrop-blur-2xl header-glow">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Shield className="text-white/70" size={26} />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white rounded-full animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              <span className="text-white font-rajdhani">
                AEGIS
              </span>
            </h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-[0.2em] -mt-0.5">
              Financial Fraud Detection & Governance
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Clock */}
          <div className="flex items-center gap-1.5 text-gray-400 text-xs font-mono">
            <Clock size={11} />
            {currentTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>

          {/* Status Indicators */}
          <div className="flex items-center gap-2">
            {/* Demo toggle */}
            <button
              onClick={toggleDemo}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border transition-all ${
                demoRunning
                  ? 'bg-red-900/40 text-red-400 border-red-500/40 hover:bg-red-800/50'
                  : 'bg-white/5 text-white/70 border-white/15 hover:bg-white/10'
              }`}
            >
              <Zap size={12} />
              {demoRunning ? 'STOP DEMO' : 'START DEMO'}
            </button>

            {/* Fullscreen toggle */}
            <button
              onClick={toggleFullscreen}
              className="flex items-center gap-1.5 px-2 py-1 rounded-full text-xs bg-gray-800/50 text-gray-400 border border-gray-700/30 hover:bg-white/5 hover:text-white/70 hover:border-white/10 transition-all"
              title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {isFullscreen ? <Minimize size={12} /> : <Maximize size={12} />}
              {isFullscreen ? 'EXIT' : 'FULLSCREEN'}
            </button>

            {/* Nessie API status */}
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs ${
              nessieConnected
                ? 'bg-white/5 text-white/70 border border-white/10'
                : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
            }`}>
              <Database size={10} />
              {nessieConnected ? 'NESSIE' : 'MOCK'}
            </div>

            {/* Infrastructure status button */}
            <button
              onClick={() => setShowStatus(true)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border transition-all cursor-pointer ${
                allOnline
                  ? 'bg-white/5 text-white/70 border-white/10 hover:bg-white/8'
                  : 'bg-red-900/30 text-red-400 border-red-500/30 hover:bg-red-900/50'
              }`}
            >
              <Server size={10} />
              <span className="font-mono">{serviceCount.online}/{serviceCount.total}</span>
              <span className="hidden sm:inline">STATUS</span>
            </button>

            {/* Connection status */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs ${
              isConnected
                ? 'bg-white/5 text-white/70 border border-white/10'
                : 'bg-red-900/30 text-red-400 border border-red-500/30'
            }`}>
              {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
              {isConnected ? (
                <span className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                  LIVE
                </span>
              ) : 'DISCONNECTED'}
            </div>
          </div>
        </div>
      </header>

      {/* Dashboard Grid */}
      <BentoGrid>
        {/* Row 1: Stats strip */}
        <StatsOverview stats={stats} />

        {/* Row 2-3 col 1: Transaction feed (row-span-2) */}
        <TransactionFeed transactions={transactions} />

        {/* Row 2: Alerts + Disputes + Account Activity */}
        <AlertPanel alerts={alerts} agentTraces={agentTraces} activeInvestigationIds={activeInvestigationIds} />
        <DisputePanel disputes={disputes} />
        <AccountActivity transactions={transactions} />

        {/* Row 3: Agent console (col-span-2) + Network graph */}
        <AgentConsole traces={agentTraces} />
        <NetworkGraph />
      </BentoGrid>

      {/* Status Modal Overlay */}
      {showStatus && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowStatus(false);
          }}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

          {/* Modal */}
          <div className="relative w-full max-w-lg max-h-[80vh] rounded-2xl border border-gray-700/40 overflow-hidden shadow-2xl shadow-black/50"
            style={{
              background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.95) 0%, rgba(10, 10, 18, 0.95) 100%)',
            }}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700/30">
              <div className="flex items-center gap-3">
                <Server size={18} className="text-white/70" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-200">
                  Infrastructure Status
                </h2>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${
                  allOnline
                    ? 'bg-white/5 text-white/70 border-white/10'
                    : 'bg-red-900/30 text-red-400 border-red-500/30'
                }`}>
                  {serviceCount.online}/{serviceCount.total} ONLINE
                </span>
              </div>
              <button
                onClick={() => setShowStatus(false)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700/40 transition-all"
              >
                <X size={16} />
              </button>
            </div>

            {/* Modal body */}
            <div className="p-6">
              <InfraStatus />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DashboardApp;
