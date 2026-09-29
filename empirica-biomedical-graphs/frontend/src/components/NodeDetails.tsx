import { X, Link as LinkIcon, Share2 } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { ENTITY_COLORS, ENTITY_LABELS } from '@/types';

export function NodeDetails() {
  const { selectedNode, setSelectedNode, filteredGraphData } = useStore();

  if (!selectedNode || !filteredGraphData) return null;

  // Find connected nodes - handle both string IDs and object references
  const getNodeId = (node: any): string => {
    return typeof node === 'string' ? node : node.id;
  };

  const connectedEdges = filteredGraphData.edges.filter(
    (edge) => getNodeId(edge.source) === selectedNode.id || getNodeId(edge.target) === selectedNode.id
  );

  const connectedNodes = new Set<string>();
  connectedEdges.forEach((edge) => {
    const sourceId = getNodeId(edge.source);
    const targetId = getNodeId(edge.target);
    if (sourceId !== selectedNode.id) connectedNodes.add(sourceId);
    if (targetId !== selectedNode.id) connectedNodes.add(targetId);
  });

  return (
    <div className="fixed right-4 top-4 w-96 bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl border border-gray-700 z-30 max-h-[calc(100vh-2rem)] overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-gradient-to-r from-gray-900 to-gray-800 backdrop-blur-sm p-6 border-b border-gray-700 flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div
            className="w-4 h-4 rounded-full mb-2"
            style={{ backgroundColor: ENTITY_COLORS[selectedNode.group] }}
          />
          <h3 className="text-xl font-bold text-white mb-1 break-words">
            {selectedNode.id}
          </h3>
          <p className="text-sm text-gray-400">
            {ENTITY_LABELS[selectedNode.group]}
          </p>
        </div>
        <button
          onClick={() => setSelectedNode(null)}
          className="ml-4 p-2 hover:bg-gray-700 rounded-lg transition-colors flex-shrink-0"
        >
          <X className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      {/* Content */}
      <div className="p-6 space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Connections</p>
            <p className="text-2xl font-bold text-white">{selectedNode.value}</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Occurrences</p>
            <p className="text-2xl font-bold text-white">
              {selectedNode.metadata.count || 0}
            </p>
          </div>
        </div>

        {/* Connected Nodes */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
            <LinkIcon className="w-4 h-4 mr-2" />
            Connected Entities ({connectedNodes.size})
          </h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {Array.from(connectedNodes).map((nodeId) => {
              const node = filteredGraphData.nodes.find((n) => n.id === nodeId);
              if (!node) return null;

              return (
                <div
                  key={nodeId}
                  className="bg-gray-800/50 rounded-lg p-3 hover:bg-gray-800 transition-colors cursor-pointer"
                  onClick={() => setSelectedNode(node)}
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: ENTITY_COLORS[node.group] }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white font-medium truncate">
                        {node.id}
                      </p>
                      <p className="text-xs text-gray-500">
                        {ENTITY_LABELS[node.group]}
                      </p>
                    </div>
                    <span className="text-xs text-gray-500 flex-shrink-0">
                      {node.value} connections
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Relationships */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
            <Share2 className="w-4 h-4 mr-2" />
            Relationship Evidence
          </h4>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {connectedEdges.slice(0, 5).map((edge, idx) => {
              const sourceId = getNodeId(edge.source);
              const targetId = getNodeId(edge.target);
              const otherNode = sourceId === selectedNode.id ? targetId : sourceId;

              return (
                <div
                  key={idx}
                  className="bg-gray-800/50 rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-primary-400">
                      → {otherNode}
                    </p>
                    {edge.metadata?.relationship_type && (
                      <span className="text-xs px-2 py-1 bg-primary-500/20 text-primary-300 rounded">
                        {edge.metadata.relationship_type}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    {edge.title}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

