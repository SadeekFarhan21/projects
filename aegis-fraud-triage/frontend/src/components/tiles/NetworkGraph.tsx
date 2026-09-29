import React, { useState, useEffect, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Handle,
  useNodesState,
  useEdgesState,
  Position,
} from '@xyflow/react';
import type { Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Tile } from '../layout/Tile';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { API_URL } from '../../config';
import { formatCurrency } from '../../utils/formatters';
import { GraphNetworkResponse } from '../../types/events';

/* ── Status colors ── */
const STATUS_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  blocked:         { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.4)',  text: '#fca5a5' },
  awaiting_review: { bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.35)', text: '#fcd34d' },
  flagged:         { bg: 'rgba(245, 158, 11, 0.08)', border: 'rgba(245, 158, 11, 0.25)', text: '#fde68a' },
};

const DEFAULT_COLORS = { bg: 'rgba(31, 41, 55, 0.6)', border: 'rgba(75, 85, 99, 0.3)', text: '#9ca3af' };

function getStatusColors(status?: string) {
  return STATUS_COLORS[status || ''] || DEFAULT_COLORS;
}

/* ── Custom Node: Flagged Account (circle with status) ── */
function AccountNode({ data }: { data: Record<string, unknown> }) {
  const status = data.status as string;
  const label = data.label as string;
  const alertCount = data.alert_count as number;
  const totalAmount = data.total_amount as number;
  const colors = getStatusColors(status);

  return (
    <div className="relative flex flex-col items-center gap-1">
      <Handle type="source" position={Position.Right} style={{ background: 'transparent', border: 'none' }} />
      <Handle type="target" position={Position.Left} style={{ background: 'transparent', border: 'none' }} />
      <div
        className="flex items-center justify-center rounded-full text-[10px] font-mono font-bold border-2"
        style={{
          width: 58,
          height: 58,
          backgroundColor: colors.bg,
          borderColor: colors.border,
          color: colors.text,
        }}
      >
        {label}
      </div>
      <div className="text-[8px] text-center leading-tight" style={{ color: colors.text }}>
        <div className="font-bold uppercase" style={{ fontSize: 7 }}>{status?.replace('_', ' ')}</div>
        <div>{alertCount} alert{alertCount !== 1 ? 's' : ''} · {formatCurrency(totalAmount)}</div>
      </div>
    </div>
  );
}

/* ── Custom Node: Merchant (rectangle with fraud details) ── */
function MerchantNode({ data }: { data: Record<string, unknown> }) {
  const label = data.label as string;
  const merchantId = data.merchant_id as string;
  const category = data.category as string;
  const city = data.city as string;
  const country = data.country as string;
  const fraudCount = data.fraud_count as number;
  const totalFraud = data.total_fraud_amount as number;
  const isRing = data.is_ring_node as boolean;

  const borderColor = isRing ? 'rgba(239, 68, 68, 0.5)' : 'rgba(75, 85, 99, 0.4)';
  const bgColor = isRing ? 'rgba(239, 68, 68, 0.08)' : 'rgba(31, 41, 55, 0.6)';

  return (
    <div
      className="relative rounded-lg border px-2.5 py-1.5 text-center"
      style={{
        minWidth: 100,
        maxWidth: 140,
        backgroundColor: bgColor,
        borderColor: borderColor,
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: 'transparent', border: 'none' }} />
      <Handle type="source" position={Position.Right} style={{ background: 'transparent', border: 'none' }} />
      <div className="text-[9px] font-semibold text-gray-200 truncate">{label}</div>
      <div className="text-[7px] text-gray-500 font-mono truncate">{merchantId}</div>
      <div className="text-[7px] text-gray-400 truncate">
        {category}{city ? ` · ${city}` : ''}{country && country !== 'US' ? `, ${country}` : ''}
      </div>
      <div className="text-[7px] mt-0.5" style={{ color: isRing ? '#fca5a5' : '#9ca3af' }}>
        {fraudCount} fraud txn · {formatCurrency(totalFraud)}
      </div>
      {isRing && (
        <div className="text-[6px] font-bold uppercase mt-0.5 text-red-400">ring node</div>
      )}
    </div>
  );
}

const nodeTypes = { account: AccountNode, merchant: MerchantNode };

export function NetworkGraph() {
  const [graphData, setGraphData] = useState<GraphNetworkResponse | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  /* ── Poll /api/graph/network every 8s ── */
  const fetchGraph = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/graph/network`);
      const data: GraphNetworkResponse = await res.json();
      setGraphData(data);
    } catch {
      // Silently retry on next interval
    }
  }, []);

  useEffect(() => {
    fetchGraph();
    const interval = setInterval(fetchGraph, 8000);
    return () => clearInterval(interval);
  }, [fetchGraph]);

  /* ── Convert API response -> React Flow nodes/edges ── */
  useEffect(() => {
    if (!graphData || graphData.nodes.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }

    const accountNodes = graphData.nodes.filter(n => n.type === 'account');
    const merchantNodes = graphData.nodes.filter(n => n.type === 'merchant');

    const ACCOUNT_SPACING = 105;
    const MERCHANT_SPACING = 110;
    const LEFT_X = 0;
    const RIGHT_X = 300;

    // Center both columns vertically
    const accountHeight = Math.max(0, (accountNodes.length - 1) * ACCOUNT_SPACING);
    const merchantHeight = Math.max(0, (merchantNodes.length - 1) * MERCHANT_SPACING);
    const maxHeight = Math.max(accountHeight, merchantHeight);
    const accountOffset = (maxHeight - accountHeight) / 2;
    const merchantOffset = (maxHeight - merchantHeight) / 2;

    const flowNodes: Node[] = [
      ...accountNodes.map((n, i) => ({
        id: n.id,
        type: 'account' as const,
        position: { x: LEFT_X, y: i * ACCOUNT_SPACING + accountOffset },
        data: {
          label: n.label,
          status: n.status,
          alert_count: n.alert_count ?? 0,
          total_amount: n.total_amount ?? 0,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      })),
      ...merchantNodes.map((n, i) => ({
        id: n.id,
        type: 'merchant' as const,
        position: { x: RIGHT_X, y: i * MERCHANT_SPACING + merchantOffset },
        data: {
          label: n.label,
          merchant_id: n.merchant_id ?? n.id,
          category: n.category ?? 'Unknown',
          city: n.city ?? '',
          country: n.country ?? '',
          fraud_count: n.fraud_count ?? 0,
          total_fraud_amount: n.total_fraud_amount ?? 0,
          is_ring_node: n.is_ring_node ?? false,
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      })),
    ];

    const flowEdges: Edge[] = graphData.edges.map((e, i) => {
      const isBlocked = e.status === 'blocked';
      const isReview = e.status === 'awaiting_review';
      return {
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        animated: isBlocked || isReview,
        label: formatCurrency(e.amount),
        labelStyle: { fill: '#6b7280', fontSize: 8, fontFamily: 'monospace' },
        labelBgStyle: { fill: 'rgba(10, 10, 15, 0.8)', rx: 4 },
        labelBgPadding: [4, 2] as [number, number],
        style: {
          stroke: isBlocked
            ? 'rgba(239, 68, 68, 0.5)'
            : isReview
              ? 'rgba(245, 158, 11, 0.4)'
              : 'rgba(245, 158, 11, 0.25)',
          strokeWidth: isBlocked ? 2 : 1.5,
        },
      };
    });

    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [graphData, setNodes, setEdges]);

  /* ── Badge ── */
  const fraudCount = graphData ? graphData.total_accounts : 0;
  const ringCount = graphData ? graphData.ring_merchants : 0;
  const badge = graphData && graphData.nodes.length > 0 ? (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-900/20 text-red-400 border border-red-500/20 font-mono">
        {fraudCount} flagged
      </span>
      {ringCount > 0 && (
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-red-900/30 text-red-300 border border-red-500/30 font-mono">
          {ringCount} ring
        </span>
      )}
    </div>
  ) : null;

  return (
    <Tile title="Fraud Network" accentColor="#ef4444" badge={badge}>
      <div className="h-full w-full">
        {!graphData ? (
          <LoadingSpinner label="Loading network..." />
        ) : graphData.nodes.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-1">
            <div className="text-gray-600 text-xs">No fraud detected yet</div>
            <div className="text-gray-700 text-[10px]">Blocked and flagged transactions appear here</div>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            proOptions={{ hideAttribution: true }}
            panOnDrag
            zoomOnScroll
            minZoom={0.2}
            maxZoom={2.5}
            style={{ background: 'transparent' }}
          >
            <Background color="rgba(75, 85, 99, 0.06)" gap={24} />
          </ReactFlow>
        )}
      </div>
    </Tile>
  );
}
