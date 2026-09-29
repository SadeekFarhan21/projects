import { useState } from 'react';
import { MessageCircle, Bot, Send, Clock, Sparkles } from 'lucide-react';
import { apiService } from '@/services/api';
import { useStore } from '@/store/useStore';

type Message = { role: 'user' | 'assistant'; content: string; isAgentic?: boolean; researchId?: string };

// Simple markdown rendering function
const renderMarkdown = (text: string) => {
  // Split by lines
  const lines = text.split('\n');
  const elements: JSX.Element[] = [];
  
  lines.forEach((line, idx) => {
    // Bold text: **text**
    let content = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic text: *text*
    content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Code: `text`
    content = content.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Headers
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={idx} className="text-lg font-bold mt-2 mb-1">{line.substring(4)}</h3>
      );
    } else if (line.startsWith('## ')) {
      elements.push(
        <h2 key={idx} className="text-xl font-bold mt-3 mb-2">{line.substring(3)}</h2>
      );
    } else if (line.startsWith('# ')) {
      elements.push(
        <h1 key={idx} className="text-2xl font-bold mt-4 mb-2">{line.substring(2)}</h1>
      );
    } 
    // Bullet points
    else if (line.startsWith('• ') || line.startsWith('- ')) {
      elements.push(
        <div key={idx} className="ml-4" dangerouslySetInnerHTML={{ __html: '• ' + content.substring(2) }} />
      );
    }
    // Numbered list
    else if (/^\d+\. /.test(line)) {
      elements.push(
        <div key={idx} className="ml-4" dangerouslySetInnerHTML={{ __html: line }} />
      );
    }
    // Empty line
    else if (line.trim() === '') {
      elements.push(<div key={idx} className="h-2" />);
    }
    // Regular text with inline formatting
    else {
      elements.push(
        <div key={idx} dangerouslySetInnerHTML={{ __html: content }} />
      );
    }
  });
  
  return elements;
};

export const ChatPanel = () => {
  const { filteredGraphData, setHighlightedNodes, setHighlightedLinks, currentProject } = useStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Agentic AI state
  const [agentMode, setAgentMode] = useState(false);
  const [currentResearch, setCurrentResearch] = useState<any>(null);
  const [isResearching, setIsResearching] = useState(false);

  // Check if the query should trigger agentic research
  const checkIfShouldResearch = async (query: string) => {
    const researchKeywords = [
      'research', 'find papers', 'discover', 'explore', 'investigate', 'analyze',
      'what is known about', 'tell me about', 'study', 'literature', 'papers on',
      'recent developments', 'current research', 'latest findings'
    ];
    
    const queryLower = query.toLowerCase();
    const hasKeywords = researchKeywords.some(keyword => queryLower.includes(keyword));
    
    // Auto-trigger for specific patterns
    const autoTriggerPatterns = [
      /what is known about/i,
      /find papers on/i,
      /research on/i,
      /tell me about.*research/i,
      /what are the latest.*findings/i
    ];
    
    const autoTrigger = autoTriggerPatterns.some(pattern => pattern.test(query));
    
    return {
      shouldResearch: hasKeywords || autoTrigger,
      autoTrigger,
      confidence: hasKeywords ? 0.8 : 0.6
    };
  };

  // Start agentic research
  const startAgenticResearch = async (query: string, _researchInfo: any) => {
    setIsResearching(true);
    
    // Add research message
    const researchMessage = {
      role: 'assistant' as const,
      content: `Starting research on: "${query}". This may take a few minutes...`,
      isAgentic: true
    };
    
    setMessages(prev => [...prev, researchMessage]);
    
    try {
      // Start research using the API service with proper auth
      const response = await apiService.startAgenticResearch({
        research_topic: query,
        max_papers: 10,
        search_strategy: 'comprehensive'
      });
      
      const data = response;
      setCurrentResearch(data);
      
      // Poll for results
      pollResearchStatus(data.research_id);
      
    } catch (error) {
      console.error('Failed to start research:', error);
      setMessages(prev => [...prev, {
        role: 'assistant' as const,
        content: 'Sorry, I couldn\'t start the research. Please try again.',
        isAgentic: true
      }]);
      setIsResearching(false);
    }
  };

  // Poll research status
  const pollResearchStatus = async (researchId: string) => {
    const poll = async () => {
      try {
        const status = await apiService.getAgenticResearchStatus(researchId);
        
        // Update current research with latest status
        setCurrentResearch(status);
        
        if (status.status === 'completed') {
          // Get results
          const results = await apiService.getAgenticResearchResults(researchId);
          
          // Format results for display
          const resultMessage = formatResearchResults(results);
          setMessages(prev => [...prev, {
            role: 'assistant' as const,
            content: resultMessage,
            isAgentic: true,
            researchId
          }]);
          
          setIsResearching(false);
          setCurrentResearch(null);
        } else if (status.status === 'failed') {
          setMessages(prev => [...prev, {
            role: 'assistant' as const,
            content: `Research failed: ${status.error}`,
            isAgentic: true
          }]);
          setIsResearching(false);
          setCurrentResearch(null);
        } else {
          // Still processing, poll again in 1 second for more frequent updates
          setTimeout(poll, 1000);
        }
      } catch (error) {
        console.error('Failed to check research status:', error);
        setIsResearching(false);
      }
    };
    
    poll();
  };

  // Format research results for display
  const formatResearchResults = (results: any) => {
    const papers = results.papers_analyzed || 0;
    const insights = results.insights || [];
    const recommendations = results.recommendations || {};
    
    let message = `**Research Complete!**\n\n`;
    message += `**Papers Analyzed:** ${papers}\n\n`;
    
    if (insights.length > 0) {
      message += `**Key Insights:**\n`;
      insights.slice(0, 3).forEach((insight: any, i: number) => {
        message += `${i + 1}. **${insight.title}**\n`;
        message += `   ${insight.description?.substring(0, 100)}...\n\n`;
      });
    }
    
    if (recommendations.research_gaps?.length > 0) {
      message += `**Research Gaps Identified:**\n`;
      recommendations.research_gaps.slice(0, 2).forEach((gap: string, i: number) => {
        message += `${i + 1}. ${gap}\n`;
      });
      message += `\n`;
    }
    
    message += `**Knowledge Graph:** ${results.knowledge_graph?.nodes?.length || 0} entities, ${results.knowledge_graph?.edges?.length || 0} relationships`;
    
    return message;
  };

  const send = async () => {
    if (!input.trim()) return;
    
    // Clear highlights from previous query
    setHighlightedNodes(new Set());
    setHighlightedLinks(new Set());
    
    const nextHistory = [...messages, { role: 'user' as const, content: input }];
    setMessages(nextHistory);
    setLoading(true);
    setInput(''); // Clear input immediately
    
    try {
      // Check if we should trigger agentic research
      const shouldResearch = await checkIfShouldResearch(input);
      
      if (shouldResearch && (agentMode || shouldResearch.autoTrigger)) {
        await startAgenticResearch(input, shouldResearch);
        return;
      }
      
      // Regular chat if we have graph data
      if (filteredGraphData) {
        const res = await apiService.chat(input, filteredGraphData, nextHistory, currentProject?.project_id);
        setMessages([...nextHistory, { role: 'assistant' as const, content: res.answer }]);

        // Highlight relevant nodes and edges
        const nodeSet = new Set<string>(res.relevant_nodes || []);
        const linkSet = new Set<string>((res.relevant_edges || []).map(([a, b]) => `${a}-${b}`));
        setHighlightedNodes(nodeSet);
        setHighlightedLinks(linkSet);
      } else {
        // No graph data available
        setMessages([...nextHistory, { 
          role: 'assistant' as const, 
          content: 'No graph data available. Please upload and process some PDFs first, or try asking me to research a topic!' 
        }]);
      }
    } catch (e) {
      setMessages([...nextHistory, { role: 'assistant' as const, content: 'Error answering. Check backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed right-0 top-0 h-full w-[480px] bg-gradient-to-b from-gray-900/98 via-gray-900/98 to-gray-800/98 border-l-2 border-purple-500/30 backdrop-blur-xl z-20 flex flex-col shadow-2xl">
      {/* Header */}
      <div className="p-4 border-b border-gray-700/50 bg-gradient-to-r from-purple-900/20 to-blue-900/20">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <h2 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              Chat
            </h2>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setAgentMode(!agentMode)}
              className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all flex items-center space-x-1 ${
                agentMode 
                  ? 'bg-gradient-to-r from-green-600 to-green-500 text-white shadow-lg shadow-green-500/30' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Bot className="w-3.5 h-3.5" />
              <span>{agentMode ? 'Agent ON' : 'Agent OFF'}</span>
            </button>
            
            {messages.length > 0 && (
              <button 
                onClick={() => {
                  setMessages([]);
                  setHighlightedNodes(new Set());
                  setHighlightedLinks(new Set());
                }} 
                className="text-xs px-3 py-1.5 bg-red-600/20 hover:bg-red-600/30 rounded-lg text-red-400 font-medium transition-all border border-red-500/30"
              >
                Clear
              </button>
            )}
          </div>
        </div>
        
        {/* Agent Mode Description */}
        <p className="text-xs text-gray-400 flex items-center">
          {agentMode ? (
            <>
              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
              AI-powered research with automatic paper discovery
            </>
          ) : (
            <>
              <MessageCircle className="w-3.5 h-3.5 mr-1.5" />
              Chat about your knowledge graph
            </>
          )}
        </p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-auto p-4 space-y-4 pb-40 custom-scrollbar">
        {messages.map((m, i) => (
          <div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div className={`inline-block px-4 py-3 rounded-2xl max-w-[85%] ${
              m.role === 'user' 
                ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20' 
                : m.isAgentic 
                  ? 'bg-gradient-to-r from-purple-900/80 to-purple-800/80 text-purple-50 border border-purple-500/30 shadow-lg shadow-purple-500/10' 
                  : 'bg-gray-800/80 text-gray-100 border border-gray-700/50'
            }`}>
              {m.isAgentic && (
                <div className="text-xs text-purple-300 mb-2 font-semibold flex items-center">
                  <Bot className="w-3.5 h-3.5 mr-1.5" />
                  <span>Agentic Research</span>
                </div>
              )}
              <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                {renderMarkdown(m.content)}
              </div>
            </div>
          </div>
        ))}
        
        {messages.length === 0 && (
          <div className="text-center py-8 space-y-4">
            <MessageCircle className="w-16 h-16 mx-auto text-gray-600 mb-4" />
            <div className="text-gray-300 font-medium">Start a conversation</div>
            <div className="text-gray-500 text-sm space-y-3 max-w-sm mx-auto">
              <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
                <div className="font-medium text-gray-400 mb-2">Graph Queries:</div>
                <div className="space-y-1 text-xs text-gray-500">
                  <div>• "shortest path between TP53 and cancer"</div>
                  <div>• "neighbors of VEGF"</div>
                  <div>• "common connections TP53, BRCA1"</div>
                </div>
              </div>
              {agentMode && (
                <div className="bg-purple-900/20 rounded-lg p-3 border border-purple-500/30">
                  <div className="font-medium text-purple-400 mb-2">Agentic Research:</div>
                  <div className="space-y-1 text-xs text-purple-300/70">
                    <div>• "research gut microbiome and immunotherapy"</div>
                    <div>• "find papers on cancer biomarkers"</div>
                    <div>• "what is known about CRISPR therapy"</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Research Status Indicator - Micro Updates */}
        {isResearching && (
          <div className="bg-gradient-to-r from-purple-900/60 to-purple-800/60 border-2 border-purple-500/40 rounded-xl p-4 space-y-3 shadow-xl shadow-purple-500/20">
            {/* Loading spinner with current stage */}
            <div className="flex items-center space-x-3">
              <div className="relative w-7 h-7">
                <div className="absolute inset-0 border-3 border-transparent border-t-purple-400 border-r-purple-400 rounded-full animate-spin"></div>
              </div>
              <span className="text-sm font-semibold text-purple-100">
                {currentResearch?.current_stage || 'Processing...'}
              </span>
            </div>
            
            {currentResearch && currentResearch.progress && (
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-purple-900/40 rounded-lg p-2 border border-purple-500/20">
                  <div className="text-purple-300 mb-1">Papers found</div>
                  <div className="text-2xl font-bold text-purple-100">{currentResearch.progress.papers_found}</div>
                </div>
                <div className="bg-purple-900/40 rounded-lg p-2 border border-purple-500/20">
                  <div className="text-purple-300 mb-1">Analyzed</div>
                  <div className="text-2xl font-bold text-purple-100">{currentResearch.progress.papers_analyzed}</div>
                </div>
                <div className="bg-purple-900/40 rounded-lg p-2 border border-purple-500/20">
                  <div className="text-purple-300 mb-1">Entities</div>
                  <div className="text-2xl font-bold text-purple-100">{currentResearch.progress.entities_extracted}</div>
                </div>
                <div className="bg-purple-900/40 rounded-lg p-2 border border-purple-500/20">
                  <div className="text-purple-300 mb-1">Relationships</div>
                  <div className="text-2xl font-bold text-purple-100">{currentResearch.progress.relationships_found}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area - Fixed at Bottom */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-gray-900 via-gray-900 to-transparent border-t border-gray-700/50">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={
              agentMode 
                ? "Ask me to research anything..." 
                : "Ask about your knowledge graph..."
            }
            className="flex-1 bg-gray-800/80 text-gray-100 px-4 py-3 rounded-xl outline-none border-2 border-gray-700 focus:border-purple-500 resize-none transition-all placeholder-gray-500"
            rows={2}
          />
          <button
            onClick={send}
            disabled={loading || isResearching}
            className={`px-6 rounded-xl text-white font-semibold transition-all shadow-lg flex items-center justify-center ${
              isResearching 
                ? 'bg-purple-600/50 cursor-not-allowed' 
                : 'bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 shadow-blue-500/30'
            } disabled:opacity-50`}
          >
            {isResearching ? (
              <Clock className="w-5 h-5 animate-pulse" />
            ) : loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        <div className="text-xs text-gray-500 mt-2 text-center">
          Press Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};


