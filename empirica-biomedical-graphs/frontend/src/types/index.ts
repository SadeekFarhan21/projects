export type EntityType =
  | 'GENE_OR_GENE_PRODUCT'
  | 'CHEMICAL'
  | 'DISEASE'
  | 'ORGANISM'
  | 'CELL_TYPE'
  | 'TISSUE'
  | 'ORGAN'
  | 'ENTITY'
  | 'UNKNOWN';

export interface Node {
  id: string;
  group: EntityType;
  value: number;
  metadata: {
    count?: number;
    degree?: number;
    [key: string]: any;
  };
}

export interface Edge {
  source: string;
  target: string;
  value: number;
  title: string;
  metadata: {
    all_evidence?: string[];
    relationship_type?: string;
    [key: string]: any;
  };
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
  metadata: {
    total_nodes?: number;
    total_edges?: number;
    density?: number;
    analytics?: GraphAnalytics;
    [key: string]: any;
  };
}

export interface GraphAnalytics {
  total_nodes: number;
  total_edges: number;
  density: number;
  avg_degree: number;
  communities: string[][];
  centrality_scores: Record<string, number>;
  entity_counts: Record<string, number>;
}

export interface PDFMetadata {
  document_id: string;
  filename: string;
  uploaded_at: string;
  processed: boolean;
  selected: boolean;
  node_count: number;
  edge_count: number;
  entity_counts: Record<string, number>;
}

export interface ProjectInfo {
  project_id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  pdf_count: number;
  pdfs: PDFMetadata[];
}

export interface ProcessingStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  result?: GraphData;
}

export interface FilterOptions {
  minDegree: number;
  entityTypes: EntityType[];
  topN?: number;
  searchQuery: string;
}

export interface ViewMode {
  dimension: '2d' | '3d';
  showLabels: boolean;
  showEdgeLabels: boolean;
}

export const ENTITY_COLORS: Record<EntityType, string> = {
  GENE_OR_GENE_PRODUCT: '#3b82f6', // Blue
  CHEMICAL: '#10b981', // Green
  DISEASE: '#ef4444', // Red
  ORGANISM: '#f59e0b', // Orange
  CELL_TYPE: '#8b5cf6', // Purple
  TISSUE: '#ec4899', // Pink
  ORGAN: '#14b8a6', // Teal
  ENTITY: '#60a5fa', // Light Blue (generic)
  UNKNOWN: '#6b7280', // Gray
};

export const ENTITY_LABELS: Record<EntityType, string> = {
  GENE_OR_GENE_PRODUCT: 'Gene/Protein',
  CHEMICAL: 'Chemical',
  DISEASE: 'Disease',
  ORGANISM: 'Organism',
  CELL_TYPE: 'Cell Type',
  TISSUE: 'Tissue',
  ORGAN: 'Organ',
  ENTITY: 'Entity',
  UNKNOWN: 'Unknown',
};

