import { useState } from 'react';
import { BarChart3, Plus, X, Sparkles } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { apiService } from '@/services/api';
import toast from 'react-hot-toast';

export function HypothesisBar() {
  const { filteredGraphData, currentProject } = useStore();
  const [hypotheses, setHypotheses] = useState<
    Array<{ title: string; explanation: string; entities: string[]; confidence: number }>
  >([]);
  const [loadingHyp, setLoadingHyp] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchHypotheses = async () => {
    if (!filteredGraphData) return;
    setLoadingHyp(true);
    try {
      const res = await apiService.generateHypotheses(
        filteredGraphData,
        undefined,
        10,
        currentProject?.project_id
      );
      setHypotheses(res.hypotheses || []);
      setIsExpanded(true);
      toast.success('Hypotheses generated!');
    } catch (e) {
      setHypotheses([]);
      toast.error('Failed to generate hypotheses');
    } finally {
      setLoadingHyp(false);
    }
  };

  const highlightHypothesis = (h: any) => {
    const nodes = new Set<string>(h.entities || []);
    const links = new Set<string>();
    // @ts-ignore
    (h.edge_pairs || []).forEach(([a, b]: [string, string]) => links.add(`${a}-${b}`));
    const es = h.entities || [];
    for (let i = 0; i < es.length; i++) {
      for (let j = i + 1; j < es.length; j++) {
        links.add(`${es[i]}-${es[j]}`);
      }
    }
    useStore.getState().setHighlightedNodes(nodes);
    useStore.getState().setHighlightedLinks(links);
    toast.success('Hypothesis highlighted!');
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-30 pointer-events-none">
      {/* Glowing bar */}
      <div className="ml-96 mr-[480px] px-4 pb-6 pointer-events-auto">
        <div
          className={`bg-gradient-to-r from-blue-900/95 via-purple-900/95 to-blue-900/95 backdrop-blur-xl rounded-2xl border-2 border-blue-400/50 shadow-2xl transition-all duration-300 ${
            isExpanded ? 'shadow-blue-500/50' : 'shadow-blue-500/30'
          }`}
          style={{
            boxShadow: '0 0 40px rgba(59, 130, 246, 0.4), 0 0 80px rgba(139, 92, 246, 0.2)',
          }}
        >
          {/* Header Bar */}
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Sparkles className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white flex items-center">
                  Discovery Lab
                  {hypotheses.length > 0 && (
                    <span className="ml-2 px-2 py-0.5 bg-blue-500/30 rounded-full text-xs">
                      {hypotheses.length}
                    </span>
                  )}
                </h3>
                <p className="text-xs text-gray-300">
                  AI-powered insights from your knowledge graph
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={fetchHypotheses}
                disabled={!filteredGraphData || loadingHyp}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-all flex items-center space-x-2"
              >
                {loadingHyp ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    <span>Generate</span>
                  </>
                )}
              </button>
              {hypotheses.length > 0 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                  {isExpanded ? (
                    <X className="w-5 h-5 text-white" />
                  ) : (
                    <BarChart3 className="w-5 h-5 text-white" />
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Expanded Content */}
          {isExpanded && hypotheses.length > 0 && (
            <div className="px-4 pb-4 max-h-80 overflow-y-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {hypotheses.map((h, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-gray-900/80 rounded-lg border border-gray-700 cursor-pointer hover:bg-gray-800 hover:border-blue-500/50 transition-all"
                    onClick={() => highlightHypothesis(h)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="text-sm text-white font-bold flex-1">{h.title}</div>
                      <div className="ml-2 px-2 py-0.5 bg-blue-600/30 border border-blue-500/50 rounded text-xs text-blue-300 font-medium">
                        {(h.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="text-xs text-gray-300 leading-relaxed line-clamp-2">
                      {h.explanation}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {h.entities.slice(0, 3).map((entity, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-0.5 bg-gray-800 rounded text-blue-400 border border-gray-700"
                        >
                          {entity}
                        </span>
                      ))}
                      {h.entities.length > 3 && (
                        <span className="text-xs px-2 py-0.5 bg-gray-800 rounded text-gray-500">
                          +{h.entities.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {isExpanded && hypotheses.length === 0 && (
            <div className="px-4 pb-4 text-center py-8">
              <Sparkles className="w-12 h-12 mx-auto mb-2 text-blue-400 opacity-50" />
              <p className="text-sm text-gray-300">No hypotheses yet</p>
              <p className="text-xs text-gray-500">Click Generate to discover insights</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
