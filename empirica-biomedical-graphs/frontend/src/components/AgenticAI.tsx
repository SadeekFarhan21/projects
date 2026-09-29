import React, { useState, useEffect } from 'react';
import { BookOpen, Target, BarChart, Rocket, Search, Brain, CheckCircle, XCircle, Clock, Bot } from 'lucide-react';
import { apiService } from '@/services/api';
import { useStore } from '@/store/useStore';

interface AgenticResearch {
  research_id: string;
  status: 'starting' | 'searching' | 'analyzing' | 'completed' | 'failed' | string;
  progress: {
    papers_found: number;
    papers_analyzed: number;
    entities_extracted: number;
    relationships_found: number;
  };
  results?: any;
  error?: string;
}

interface AgenticAIProps {
  onResearchComplete?: (results: any) => void;
}

export const AgenticAI: React.FC<AgenticAIProps> = ({ onResearchComplete }) => {
  const { currentProject } = useStore();
  const [researchTopic, setResearchTopic] = useState('');
  const [maxPapers, setMaxPapers] = useState(10);
  const [searchStrategy, setSearchStrategy] = useState('comprehensive');
  const [currentResearch, setCurrentResearch] = useState<AgenticResearch | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const fetchResults = async (researchId: string) => {
    try {
      const results = await apiService.getAgenticResearchResults(researchId);
      setResults(results);
      if (onResearchComplete) {
        onResearchComplete(results);
      }
    } catch (error) {
      console.error('Failed to fetch results:', error);
    }
  };

  const startResearch = async () => {
    if (!researchTopic.trim()) {
      alert('Please enter a research topic');
      return;
    }

    setIsLoading(true);
    try {
      const data = await apiService.startAgenticResearch({
        research_topic: researchTopic,
        max_papers: maxPapers,
        search_strategy: searchStrategy,
        project_id: currentProject?.project_id
      });
      
      setCurrentResearch({
        research_id: data.research_id,
        status: 'starting',
        progress: {
          papers_found: 0,
          papers_analyzed: 0,
          entities_extracted: 0,
          relationships_found: 0,
        },
      });
    } catch (error) {
      console.error('Failed to start research:', error);
      alert('Failed to start research');
    } finally {
      setIsLoading(false);
    }
  };

  const checkStatus = async (researchId: string) => {
    try {
      const data = await apiService.getAgenticResearchStatus(researchId);
      setCurrentResearch(data);

      if (data.status === 'completed') {
        await fetchResults(researchId);
      }
    } catch (error) {
      console.error('Failed to check status:', error);
    }
  };

  useEffect(() => {
    if (currentResearch && currentResearch.status !== 'completed' && currentResearch.status !== 'failed') {
      const interval = setInterval(() => {
        checkStatus(currentResearch.research_id);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [currentResearch]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'starting': return 'text-blue-600';
      case 'searching': return 'text-yellow-600';
      case 'analyzing': return 'text-purple-600';
      case 'completed': return 'text-green-600';
      case 'failed': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'starting': return <Rocket className="w-4 h-4" />;
      case 'searching': return <Search className="w-4 h-4" />;
      case 'analyzing': return <Brain className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <XCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <Bot className="w-6 h-6" />
          Agentic AI Research
        </h2>
        
        {!currentResearch && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Research Topic
              </label>
              <input
                type="text"
                value={researchTopic}
                onChange={(e) => setResearchTopic(e.target.value)}
                placeholder="e.g., gut microbiome and cancer immunotherapy"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Papers
                </label>
                <input
                  type="number"
                  value={maxPapers}
                  onChange={(e) => setMaxPapers(parseInt(e.target.value))}
                  min="1"
                  max="50"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Search Strategy
                </label>
                <select
                  value={searchStrategy}
                  onChange={(e) => setSearchStrategy(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="comprehensive">Comprehensive</option>
                  <option value="recent">Recent</option>
                  <option value="high_impact">High Impact</option>
                </select>
              </div>
            </div>
            
            <button
              onClick={startResearch}
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Starting Research...' : 'Start Autonomous Research'}
            </button>
          </div>
        )}

        {currentResearch && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-800">
                Research Status
              </h3>
              <span className={`flex items-center space-x-2 ${getStatusColor(currentResearch.status)}`}>
                <span>{getStatusIcon(currentResearch.status)}</span>
                <span className="capitalize">{currentResearch.status}</span>
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-3 rounded-md">
                <div className="text-sm text-gray-600">Papers Found</div>
                <div className="text-lg font-semibold">{currentResearch.progress.papers_found}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-md">
                <div className="text-sm text-gray-600">Papers Analyzed</div>
                <div className="text-lg font-semibold">{currentResearch.progress.papers_analyzed}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-md">
                <div className="text-sm text-gray-600">Entities Extracted</div>
                <div className="text-lg font-semibold">{currentResearch.progress.entities_extracted}</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-md">
                <div className="text-sm text-gray-600">Relationships Found</div>
                <div className="text-lg font-semibold">{currentResearch.progress.relationships_found}</div>
              </div>
            </div>
            
            {currentResearch.error && (
              <div className="bg-red-50 border border-red-200 rounded-md p-3">
                <div className="text-red-800 font-medium">Error:</div>
                <div className="text-red-600">{currentResearch.error}</div>
              </div>
            )}
            
            {currentResearch.status === 'completed' && (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-md p-3">
                  <div className="text-green-800 font-medium">Research Completed!</div>
                  <div className="text-green-600">
                    {results ? (
                      <div className="space-y-2">
                        <div className="font-semibold">Research Complete!</div>
                        <div className="flex items-center gap-2">
                          <BookOpen className="w-4 h-4" />
                          <span>Analyzed {results.papers_analyzed} papers</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Target className="w-4 h-4" />
                          <span>Research Gaps Identified: {results.recommendations?.research_gaps?.length || 0} gaps found</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <BarChart className="w-4 h-4" />
                          <span>Knowledge Graph: {results.knowledge_graph?.nodes?.length || 0} entities, {results.knowledge_graph?.edges?.length || 0} relationships</span>
                        </div>
                        <div className="mt-2 font-medium">Auto-Saving: PDFs are being downloaded and saved as a new project automatically!</div>
                      </div>
                    ) : (
                      `Analyzed ${currentResearch.progress.papers_analyzed} papers and found ${currentResearch.progress.entities_extracted} entities.`
                    )}
                  </div>
                </div>
                
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setCurrentResearch(null);
                      setResults(null);
                    }}
                    className="bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700"
                  >
                    Start New Research
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {results && (
          <div className="mt-6 space-y-4">
            <h3 className="text-lg font-semibold text-gray-800">Research Results</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-blue-50 p-4 rounded-md">
                <h4 className="font-medium text-blue-800 mb-2">Papers Analyzed</h4>
                <div className="text-2xl font-bold text-blue-600">{results.papers_analyzed}</div>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-md">
                <h4 className="font-medium text-purple-800 mb-2">Knowledge Graph</h4>
                <div className="text-sm text-purple-600">
                  {results.knowledge_graph?.nodes?.length || 0} nodes, {results.knowledge_graph?.edges?.length || 0} edges
                </div>
              </div>
            </div>
            
            {results.insights && results.insights.length > 0 && (
              <div className="bg-yellow-50 p-4 rounded-md">
                <h4 className="font-medium text-yellow-800 mb-2">Key Insights</h4>
                <div className="space-y-2">
                  {results.insights.slice(0, 3).map((insight: any, index: number) => (
                    <div key={index} className="text-sm text-yellow-700">
                      <div className="font-medium">{insight.title}</div>
                      <div className="text-gray-600">{insight.description?.substring(0, 100)}...</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {results.recommendations && (
              <div className="bg-green-50 p-4 rounded-md">
                <h4 className="font-medium text-green-800 mb-2">Research Recommendations</h4>
                <div className="text-sm text-green-700">
                  {results.recommendations.research_gaps?.length || 0} research gaps identified
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
