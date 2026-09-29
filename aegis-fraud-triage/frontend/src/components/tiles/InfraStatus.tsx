import React, { useState, useEffect } from 'react';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { API_URL } from '../../config';
import {
  Server,
  GitBranch,
  Zap,
  Database,
  CheckCircle,
  XCircle,
  Cpu,
  Search,
} from 'lucide-react';

interface HealthData {
  status: string;
  nessie_connected: boolean;
  anomaly_engine: boolean;
  producer_active: boolean;
  vector_store_enabled: boolean;
  neo4j_connected: boolean;
  redis_connected: boolean;
}

function StatusRow({
  icon,
  label,
  connected,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  connected: boolean;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between py-1.5 px-2.5 rounded-lg bg-gray-800/20 border border-gray-700/15">
      <div className="flex items-center gap-2">
        <span style={{ color: connected ? color : '#4b5563' }}>{icon}</span>
        <span className="text-xs text-gray-300 font-medium">{label}</span>
      </div>
      <div className="flex items-center gap-1.5">
        {connected ? (
          <>
            <CheckCircle size={12} className="text-white/70" />
            <span className="text-[10px] text-white/70 font-mono tracking-wider">
              ONLINE
            </span>
          </>
        ) : (
          <>
            <XCircle size={12} className="text-gray-500" />
            <span className="text-[10px] text-gray-500 font-mono tracking-wider">
              OFFLINE
            </span>
          </>
        )}
      </div>
    </div>
  );
}

export function InfraStatus() {
  const [health, setHealth] = useState<HealthData | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/api/health`);
        const data = await res.json();
        setHealth(data);
      } catch {
        setHealth(null);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const services = health
    ? [
        {
          icon: <Database size={14} />,
          label: 'Nessie API',
          connected: health.nessie_connected,
          color: '#ffffff',
        },
        {
          icon: <Cpu size={14} />,
          label: 'Anomaly Engine',
          connected: health.anomaly_engine,
          color: '#d1d5db',
        },
        {
          icon: <Search size={14} />,
          label: 'FAISS Vector Store',
          connected: health.vector_store_enabled,
          color: '#d1d5db',
        },
        {
          icon: <GitBranch size={14} />,
          label: 'Neo4j Graph DB',
          connected: health.neo4j_connected,
          color: '#d1d5db',
        },
        {
          icon: <Zap size={14} />,
          label: 'Redis Cache',
          connected: health.redis_connected,
          color: '#d1d5db',
        },
        {
          icon: <Server size={14} />,
          label: 'Redpanda Kafka',
          connected: health.producer_active,
          color: '#d1d5db',
        },
      ]
    : [];

  const onlineCount = services.filter((s) => s.connected).length;

  const badge = health ? (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border ${
        onlineCount === services.length
          ? 'bg-white/5 text-white/70 border-white/10'
          : 'bg-red-900/30 text-red-400 border-red-500/30'
      }`}
    >
      {onlineCount}/{services.length}
    </span>
  ) : null;

  return (
    <Tile
      title="Infrastructure"
      accentColor="#ea580c"
      className="h-full"
      badge={badge}
    >
      <div className="space-y-1.5 overflow-y-auto h-full pr-1">
        {!health && (
          <LoadingSpinner label="Checking services..." />
        )}
        {services.map((svc, i) => (
          <StatusRow key={i} {...svc} />
        ))}
      </div>
    </Tile>
  );
}
