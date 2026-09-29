import { create } from 'zustand';
import { GraphData, Node, FilterOptions, ViewMode, ProcessingStatus, ProjectInfo, PDFMetadata } from '@/types';

interface AppState {
  // Project and PDF data
  currentProject: ProjectInfo | null;
  pdfs: PDFMetadata[];
  selectedPdfIds: Set<string>;
  
  // Graph data
  graphData: GraphData | null;
  filteredGraphData: GraphData | null;
  
  // Selected node
  selectedNode: Node | null;
  highlightedNodes: Set<string>;
  highlightedLinks: Set<string>;
  
  // Filters
  filterOptions: FilterOptions;
  
  // View settings
  viewMode: ViewMode;
  
  // Processing
  processingStatus: ProcessingStatus | null;
  isProcessing: boolean;
  
  // UI state
  sidebarOpen: boolean;
  analyticsOpen: boolean;
  showProjectSelection: boolean;
  showUploadPanel: boolean;
  isLoadingProject: boolean;
  
  // Actions
  setCurrentProject: (project: ProjectInfo | null) => void;
  setPdfs: (pdfs: PDFMetadata[]) => void;
  togglePdfSelection: (documentId: string) => void;
  setSelectedPdfIds: (ids: Set<string>) => void;
  setGraphData: (data: GraphData | null) => void;
  setFilteredGraphData: (data: GraphData | null) => void;
  setSelectedNode: (node: Node | null) => void;
  setHighlightedNodes: (nodes: Set<string>) => void;
  setHighlightedLinks: (links: Set<string>) => void;
  setFilterOptions: (options: Partial<FilterOptions>) => void;
  setViewMode: (mode: Partial<ViewMode>) => void;
  setProcessingStatus: (status: ProcessingStatus | null) => void;
  setIsProcessing: (processing: boolean) => void;
  toggleSidebar: () => void;
  toggleAnalytics: () => void;
  setShowProjectSelection: (show: boolean) => void;
  setShowUploadPanel: (show: boolean) => void;
  setIsLoadingProject: (loading: boolean) => void;
  reset: () => void;
}

const initialFilterOptions: FilterOptions = {
  minDegree: 1,
  entityTypes: [],
  topN: undefined,
  searchQuery: '',
};

const initialViewMode: ViewMode = {
  dimension: '2d',
  showLabels: true,
  showEdgeLabels: false,
};

export const useStore = create<AppState>((set) => ({
  // Initial state
  currentProject: null,
  pdfs: [],
  selectedPdfIds: new Set(),
  graphData: null,
  filteredGraphData: null,
  selectedNode: null,
  highlightedNodes: new Set(),
  highlightedLinks: new Set(),
  filterOptions: initialFilterOptions,
  viewMode: initialViewMode,
  processingStatus: null,
  isProcessing: false,
  sidebarOpen: true,
  analyticsOpen: false,
  showProjectSelection: false,
  showUploadPanel: false,
  isLoadingProject: false,

  // Actions
  setCurrentProject: (project) => set({ currentProject: project }),
  
  setPdfs: (pdfs) => {
    const selectedIds = new Set(pdfs.filter(pdf => pdf.selected).map(pdf => pdf.document_id));
    set({ pdfs, selectedPdfIds: selectedIds });
  },
  
  togglePdfSelection: (documentId) => set((state) => {
    const newSelectedIds = new Set(state.selectedPdfIds);
    if (newSelectedIds.has(documentId)) {
      newSelectedIds.delete(documentId);
    } else {
      newSelectedIds.add(documentId);
    }
    return { selectedPdfIds: newSelectedIds };
  }),
  
  setSelectedPdfIds: (ids) => set({ selectedPdfIds: ids }),
  
  setGraphData: (data) => {
    console.log('Store setGraphData called with:', {
      hasData: !!data,
      nodes: data?.nodes?.length || 0,
      edges: data?.edges?.length || 0,
      sampleNode: data?.nodes?.[0],
      sampleEdge: data?.edges?.[0]
    });
    set({ graphData: data, filteredGraphData: data });
  },
  
  setFilteredGraphData: (data) => set({ filteredGraphData: data }),
  
  setSelectedNode: (node) => {
    if (!node) {
      set({
        selectedNode: null,
        highlightedNodes: new Set(),
        highlightedLinks: new Set(),
      });
      return;
    }

    // Highlight selected node and its neighbors
    set((state) => {
      const highlightedNodes = new Set<string>([node.id]);
      const highlightedLinks = new Set<string>();

      // Find connected nodes and edges
      state.filteredGraphData?.edges.forEach((edge) => {
        if (edge.source === node.id || edge.target === node.id) {
          highlightedNodes.add(edge.source);
          highlightedNodes.add(edge.target);
          highlightedLinks.add(`${edge.source}-${edge.target}`);
        }
      });

      return {
        selectedNode: node,
        highlightedNodes,
        highlightedLinks,
      };
    });
  },
  
  setHighlightedNodes: (nodes) => set({ highlightedNodes: nodes }),
  
  setHighlightedLinks: (links) => set({ highlightedLinks: links }),
  
  setFilterOptions: (options) =>
    set((state) => ({
      filterOptions: { ...state.filterOptions, ...options },
    })),
  
  setViewMode: (mode) =>
    set((state) => ({
      viewMode: { ...state.viewMode, ...mode },
    })),
  
  setProcessingStatus: (status) => set({ processingStatus: status }),
  
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  
  toggleAnalytics: () => set((state) => ({ analyticsOpen: !state.analyticsOpen })),
  
  setShowProjectSelection: (show) => set({ showProjectSelection: show }),
  
  setShowUploadPanel: (show) => set({ showUploadPanel: show }),
  
  setIsLoadingProject: (loading) => set({ isLoadingProject: loading }),
  
  reset: () =>
    set({
      currentProject: null,
      pdfs: [],
      selectedPdfIds: new Set(),
      graphData: null,
      filteredGraphData: null,
      selectedNode: null,
      highlightedNodes: new Set(),
      highlightedLinks: new Set(),
      filterOptions: initialFilterOptions,
      processingStatus: null,
      isProcessing: false,
    }),
}));

