import { useRef, useEffect, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import SpriteText from 'three-spritetext';
import * as THREE from 'three';
import { Network } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { ENTITY_COLORS, Node as GraphNode, Edge } from '@/types';

// Relationship type colors for edges
const RELATIONSHIP_COLORS: Record<string, string> = {
  'INTERACTS_WITH': '#3b82f6',      // Blue
  'REGULATES': '#10b981',            // Green
  'INHIBITS': '#ef4444',             // Red
  'ACTIVATES': '#f59e0b',            // Orange
  'BINDS': '#8b5cf6',                // Purple
  'ASSOCIATED_WITH': '#ec4899',      // Pink
  'MODULATES': '#14b8a6',            // Teal
  'CORRELATES_WITH': '#06b6d4',      // Cyan
  'CAUSES': '#dc2626',               // Dark Red
  'TREATS': '#059669',               // Dark Green
  'LOCATED_IN': '#7c3aed',           // Violet
  'PART_OF': '#d946ef',              // Fuchsia
  'PRODUCES': '#f97316',             // Deep Orange
  'TARGETS': '#84cc16',              // Lime
  'default': '#94a3b8'               // Slate (for unknown types)
};

// Function to get edge color based on relationship type
function getEdgeColor(edge: Edge, isHighlighted: boolean, hasHighlights: boolean): string {
  if (hasHighlights) {
    return isHighlighted ? 'rgba(255, 255, 255, 0.95)' : 'rgba(100, 100, 100, 0.2)';
  }
  
  const relType = edge.metadata?.relationship_type?.toUpperCase() || 'default';
  const baseColor = RELATIONSHIP_COLORS[relType] || RELATIONSHIP_COLORS['default'];
  
  // Convert hex to rgba with higher opacity for better visibility
  const hex = baseColor.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  
  return `rgba(${r}, ${g}, ${b}, 0.85)`;
}

export function ForceGraph3DView() {
  const graphRef = useRef<any>();
  
  const {
    filteredGraphData,
    highlightedNodes,
    highlightedLinks,
    setSelectedNode,
  } = useStore();

  useEffect(() => {
    if (graphRef.current && filteredGraphData) {
      // Set camera with even more zoom out for spacious overview
      graphRef.current.cameraPosition({ z: 2000 });
      
      // Add ambient lighting for better depth perception
      const scene = graphRef.current.scene();
      if (scene) {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 100, 100);
        scene.add(directionalLight);
        
        // Add a subtle point light that follows the camera
        const pointLight = new THREE.PointLight(0x6366f1, 1, 500);
        pointLight.position.set(0, 0, 250);
        scene.add(pointLight);
      }
      
      // Increase the force simulation spacing
      const fg = graphRef.current;
      if (fg && fg.d3Force) {
        fg.d3Force('charge')?.strength(-300); // More repulsion between nodes
        fg.d3Force('link')?.distance(150); // Longer links
      }
    }
  }, [filteredGraphData]);

  const handleNodeClick = useCallback(
    (node: any) => {
      setSelectedNode(node as GraphNode);
      
      // Look at node with more distance
      if (graphRef.current) {
        const distance = 400;
        graphRef.current.cameraPosition(
          { x: node.x, y: node.y, z: node.z + distance },
          node,
          1000
        );
      }
    },
    [setSelectedNode]
  );

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  if (!filteredGraphData) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <Network className="w-24 h-24 mx-auto mb-4 text-gray-600" />
          <h2 className="text-2xl font-bold text-white mb-2">
            No Graph Data
          </h2>
          <p className="text-gray-400">
            Upload PDFs to generate a knowledge graph
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full graph-container">
      <ForceGraph3D
        ref={graphRef}
        graphData={{
          nodes: filteredGraphData.nodes,
          links: filteredGraphData.edges
        }}
        nodeId="id"
        nodeLabel={(node: any) => {
          const n = node as GraphNode;
          return `${n.id} (${n.group}) - ${n.value} connections`;
        }}
        nodeColor={(node: any) => {
          const n = node as GraphNode;
          if (highlightedNodes.size > 0) {
            return highlightedNodes.has(n.id)
              ? ENTITY_COLORS[n.group]
              : 'rgba(100, 100, 100, 0.3)';
          }
          return ENTITY_COLORS[n.group];
        }}
        nodeRelSize={6}
        nodeVal={(node: any) => {
          const n = node as GraphNode;
          return n.value || 1;
        }}
        nodeThreeObject={(node: any) => {
          const n = node as GraphNode;
          // Create a sprite text label with bigger text
          const sprite = new SpriteText(n.id);
          sprite.color = ENTITY_COLORS[n.group];
          sprite.textHeight = 8; // Bigger text for readability
          sprite.backgroundColor = 'rgba(0, 0, 0, 0.7)';
          sprite.padding = 2;
          sprite.borderRadius = 3;
          // Position the sprite with more space above the node
          (sprite as any).position.y = 25;
          return sprite;
        }}
        nodeThreeObjectExtend={true}
        linkColor={(link: any) => {
          const edge = link as Edge;
          const linkId = `${(link.source as any).id || link.source}-${(link.target as any).id || link.target}`;
          const isHighlighted = highlightedLinks.has(linkId);
          const hasHighlights = highlightedLinks.size > 0;
          
          return getEdgeColor(edge, isHighlighted, hasHighlights);
        }}
        linkWidth={(link: any) => {
          const linkId = `${(link.source as any).id || link.source}-${(link.target as any).id || link.target}`;
          if (highlightedLinks.has(linkId)) {
            return 3;
          }
          // Thinner base width - not too thick
          const edge = link as Edge;
          return Math.max(0.8, Math.min(2, (edge.value || 1) * 0.5));
        }}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={(link: any) => {
          const linkId = `${(link.source as any).id || link.source}-${(link.target as any).id || link.target}`;
          return highlightedLinks.has(linkId) ? 4 : 2;
        }}
        linkDirectionalParticleSpeed={0.005}
        linkCurvature={0.15}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        linkOpacity={0.9}
        nodeOpacity={0.95}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        enableNodeDrag={true}
        enableNavigationControls={true}
        showNavInfo={false}
        backgroundColor="rgba(10, 10, 10, 0)"
      />
    </div>
  );
}

