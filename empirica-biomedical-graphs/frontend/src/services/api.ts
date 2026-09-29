import axios from 'axios';
import { GraphData, ProcessingStatus, GraphAnalytics, ProjectInfo, PDFMetadata } from '@/types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 || error.response?.status === 403) {
      // Token expired, invalid, or not authenticated, clear auth data
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_data');
      // Optionally redirect to login or trigger re-authentication
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  /**
   * Upload PDFs and start processing
   */
  async uploadPDFs(
    files: File[],
    projectName?: string,
    enableLLM = false
  ): Promise<ProcessingStatus> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const params = new URLSearchParams();
    if (projectName) params.append('project_name', projectName);
    params.append('enable_llm', String(enableLLM));

    const response = await api.post<ProcessingStatus>(
      `/api/process?${params.toString()}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  

  /**
   * Check processing status
   */
  async getStatus(jobId: string): Promise<ProcessingStatus> {
    const response = await api.get<ProcessingStatus>(`/api/status/${jobId}`);
    return response.data;
  },

  /**
   * Filter graph data
   */
  async filterGraph(
    graphData: GraphData,
    minDegree = 1,
    entityTypes?: string[],
    topN?: number
  ): Promise<GraphData> {
    const response = await api.post<GraphData>('/api/graph/filter', graphData, {
      params: {
        min_degree: minDegree,
        entity_types: entityTypes?.join(','),
        top_n: topN,
      },
    });

    return response.data;
  },

  /**
   * Compute analytics for a graph
   */
  async computeAnalytics(graphData: GraphData): Promise<GraphAnalytics> {
    const response = await api.post<GraphAnalytics>('/api/analytics', graphData);
    return response.data;
  },

  /**
   * Chat with the graph
   */
  async chat(message: string, graph: GraphData, conversationHistory: Array<{ role: string; content: string }> = [], projectId?: string) {
    const response = await api.post('/api/chat', {
      message,
      graph,
      conversation_history: conversationHistory,
      project_id: projectId,
    });
    return response.data as {
      answer: string;
      citations: string[];
      relevant_nodes: string[];
      relevant_edges: [string, string][];
      tool_calls: string[];
    };
  },

  /**
   * Generate hypotheses
   */
  async generateHypotheses(graph: GraphData, focusEntity?: string, maxResults = 10, projectId?: string) {
    const response = await api.post('/api/hypotheses', {
      graph,
      focus_entity: focusEntity,
      max_results: maxResults,
      project_id: projectId,
    });
    return response.data as {
      hypotheses: Array<{
        title: string;
        explanation: string;
        entities: string[];
        evidence_sentences: string[];
        edge_pairs: [string, string][];
        confidence: number;
      }>;
    };
  },

  /**
   * NER preview for raw text
   */
  async nerPreview(text: string, minOccurrences = 2, returnRaw = false) {
    const response = await api.post('/api/ner/preview', {
      text,
      min_occurrences: minOccurrences,
      return_raw: returnRaw,
    });
    return response.data as {
      sentences: Array<{ sentence_id: number; sentence: string; entities: any[] }>;
      unique_entities: Record<string, { original_name: string; type: string; count: number }>;
      raw_sentences?: Array<{ sentence_id: number; sentence: string; entities: any[] }>;
    };
  },

  /**
   * List all projects
   */
  async listProjects(): Promise<ProjectInfo[]> {
    const response = await api.get<ProjectInfo[]>('/api/projects');
    return response.data;
  },

  /**
   * Get PDFs for a specific project
   */
  async getProjectPdfs(projectId: string): Promise<PDFMetadata[]> {
    const response = await api.get<PDFMetadata[]>(`/api/projects/${projectId}/pdfs`);
    return response.data;
  },

  /**
   * Update PDF selection for a project
   */
  async updatePdfSelection(projectId: string, selectedDocumentIds: string[]) {
    const response = await api.post(`/api/projects/${projectId}/select-pdfs`, selectedDocumentIds);
    return response.data;
  },

  /**
   * Add PDFs to an existing project
   */
  async addPdfsToProject(projectId: string, files: File[]): Promise<ProcessingStatus> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await api.post<ProcessingStatus>(
      `/api/projects/${projectId}/pdfs`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },

  /**
   * Delete a PDF from a project
   */
  async deletePdfFromProject(projectId: string, documentId: string) {
    const response = await api.delete(`/api/projects/${projectId}/pdfs/${documentId}`);
    return response.data;
  },

  /**
   * Get merged graph from selected PDFs
   */
  async getProjectGraph(projectId: string, selectedOnly = true): Promise<GraphData> {
    const response = await api.get<GraphData>(`/api/projects/${projectId}/graph`, {
      params: { selected_only: selectedOnly }
    });
    return response.data;
  },

  /**
   * Export project with all PDF graphs
   */
  async exportProject(projectId: string) {
    const response = await api.get(`/api/projects/${projectId}/export`);
    return response.data;
  },

  /**
   * Import project with PDF graphs
   */
  async importProject(projectData: any) {
    const response = await api.post('/api/projects/import', {
      project_data: projectData,
      merge_with_existing: false,
    });
    return response.data;
  },

  /**
   * Discover papers from PubMed
   */
  async discoverPapers(query: string, maxResults = 10) {
    const response = await api.post('/api/discover/papers', {
      query,
      max_results: maxResults,
      auto_merge: false,
      source: 'pubmed',
    });
    return response.data as {
      papers: Array<{
        id: string;
        title: string;
        abstract: string;
        authors: string[];
        journal: string;
        year?: number;
        url: string;
      }>;
      status: string;
    };
  },

  /**
   * Process discovered papers
   */
  async processDiscoveredPapers(papers: any[]) {
    const response = await api.post('/api/discover/papers/process', papers);
    return response.data;
  },

  /**
   * Discover clinical trials
   */
  async discoverTrials(condition: string, maxResults = 20) {
    const response = await api.post('/api/discover/trials', {
      condition,
      max_results: maxResults,
    });
    return response.data as {
      trials: Array<{
        nct_id: string;
        title: string;
        condition: string;
        interventions: string[];
        phase: string;
        status: string;
        sponsor: string;
        brief_summary: string;
        url: string;
      }>;
      graph?: GraphData;
    };
  },

  /**
   * Delete a project
   */
  async deleteProject(projectId: string) {
    const response = await api.delete(`/api/projects/${projectId}`);
    return response.data;
  },

  /**
   * Rename a project
   */
  async renameProject(projectId: string, newName: string) {
    const response = await api.put(`/api/projects/${projectId}`, null, {
      params: { new_name: newName }
    });
    return response.data;
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await api.get('/');
    return response.data;
  },

  /**
   * Agentic AI Research Methods
   */
  async startAgenticResearch(request: {
    research_topic: string;
    max_papers?: number;
    search_strategy?: string;
    project_id?: string;
  }): Promise<{
    research_id: string;
    status: string;
    message: string;
  }> {
    const response = await api.post('/api/agentic/research', request);
    return response.data;
  },

  async getAgenticResearchStatus(researchId: string): Promise<{
    research_id: string;
    status: string;
    current_stage: string;
    progress: {
      papers_found: number;
      papers_analyzed: number;
      entities_extracted: number;
      relationships_found: number;
    };
    started_at: string;
    error?: string;
  }> {
    const response = await api.get(`/api/agentic/research/${researchId}/status`);
    return response.data;
  },

  async getAgenticResearchResults(researchId: string): Promise<any> {
    const response = await api.get(`/api/agentic/research/${researchId}/results`);
    return response.data;
  },

  async expandAgenticResearch(researchId: string, maxNewPapers: number = 5): Promise<{
    research_id: string;
    status: string;
    message: string;
  }> {
    const response = await api.post(`/api/agentic/research/${researchId}/expand`, {
      max_new_papers: maxNewPapers
    });
    return response.data;
  },

  async saveAgenticResearch(researchId: string, projectName?: string): Promise<{
    project_id: string;
    project_name: string;
    message: string;
    papers_analyzed: number;
    documents_created: number;
    entities_count: number;
    relationships_count: number;
  }> {
    const response = await api.post(`/api/agentic/research/${researchId}/save`, {
      project_name: projectName
    });
    return response.data;
  },
};

