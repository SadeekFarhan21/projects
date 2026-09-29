import { useStore } from '@/store/useStore';
import { apiService } from '@/services/api';
import { useState, useEffect, useRef } from 'react';
import { FileText, CheckCircle2, XCircle, Loader2, Plus, Trash2 } from 'lucide-react';
import toast from 'react-hot-toast';

export const PDFSelector = () => {
  const { 
    currentProject,
    pdfs, 
    setPdfs,
    selectedPdfIds, 
    togglePdfSelection,
    setGraphData,
    setIsProcessing,
    setProcessingStatus 
  } = useStore();
  
  const [isUpdating, setIsUpdating] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load PDFs when project changes
  useEffect(() => {
    if (currentProject) {
      loadProjectPdfs();
    }
  }, [currentProject?.project_id]);

  const loadProjectPdfs = async () => {
    if (!currentProject) return;
    
    try {
      const pdfsData = await apiService.getProjectPdfs(currentProject.project_id);
      setPdfs(pdfsData);
    } catch (error) {
      console.error('Failed to load PDFs:', error);
    }
  };

  const handleTogglePdf = async (documentId: string) => {
    if (!currentProject || isUpdating) return;
    
    // Toggle in local state
    togglePdfSelection(documentId);
    
    // Get the new selection (after toggle)
    const newSelection = new Set(selectedPdfIds);
    if (newSelection.has(documentId)) {
      newSelection.delete(documentId);
    } else {
      newSelection.add(documentId);
    }
    
    // Update backend and refresh graph
    setIsUpdating(true);
    setIsProcessing(true);
    
    try {
      // Update selection in backend
      await apiService.updatePdfSelection(
        currentProject.project_id,
        Array.from(newSelection)
      );
      
      // Fetch updated graph
      const graphData = await apiService.getProjectGraph(
        currentProject.project_id,
        true
      );
      
      setGraphData(graphData);
    } catch (error) {
      console.error('Failed to update PDF selection:', error);
      // Revert on error
      togglePdfSelection(documentId);
    } finally {
      setIsUpdating(false);
      setIsProcessing(false);
    }
  };

  const handleAddPdfs = async (files: FileList | null) => {
    if (!files || files.length === 0 || !currentProject || isAdding) return;
    
    setIsAdding(true);
    setIsProcessing(true);
    const loadingToast = toast.loading(`Adding ${files.length} PDF(s)...`);
    
    try {
      // Upload files
      const filesArray = Array.from(files);
      const status = await apiService.addPdfsToProject(currentProject.project_id, filesArray);
      setProcessingStatus(status);
      
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const updatedStatus = await apiService.getStatus(status.job_id);
          setProcessingStatus(updatedStatus);
          
          if (updatedStatus.status === 'completed') {
            clearInterval(pollInterval);
            
            // Reload project PDFs
            await loadProjectPdfs();
            
            // Refresh graph
            const graphData = await apiService.getProjectGraph(
              currentProject.project_id,
              true
            );
            setGraphData(graphData);
            
            toast.success(`Added ${files.length} PDF(s) successfully!`, { id: loadingToast });
            setIsAdding(false);
            setIsProcessing(false);
          } else if (updatedStatus.status === 'failed') {
            clearInterval(pollInterval);
            toast.error(`Failed: ${updatedStatus.message}`, { id: loadingToast });
            setIsAdding(false);
            setIsProcessing(false);
          } else {
            toast.loading(
              `${updatedStatus.message} (${Math.round(updatedStatus.progress * 100)}%)`,
              { id: loadingToast }
            );
          }
        } catch (error) {
          clearInterval(pollInterval);
          toast.error('Failed to check processing status', { id: loadingToast });
          setIsAdding(false);
          setIsProcessing(false);
        }
      }, 2000);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to add PDFs', { id: loadingToast });
      setIsAdding(false);
      setIsProcessing(false);
    }
  };

  const handleDeletePdf = async (documentId: string, filename: string) => {
    if (!currentProject || isUpdating) return;
    
    if (!confirm(`Delete "${filename}" from this project?`)) {
      return;
    }
    
    setIsUpdating(true);
    const loadingToast = toast.loading(`Deleting ${filename}...`);
    
    try {
      await apiService.deletePdfFromProject(currentProject.project_id, documentId);
      
      // Reload PDFs
      await loadProjectPdfs();
      
      // Refresh graph
      const graphData = await apiService.getProjectGraph(
        currentProject.project_id,
        true
      );
      setGraphData(graphData);
      
      toast.success(`Deleted ${filename}`, { id: loadingToast });
    } catch (error) {
      console.error('Failed to delete PDF:', error);
      toast.error('Failed to delete PDF', { id: loadingToast });
    } finally {
      setIsUpdating(false);
    }
  };

  if (!currentProject) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">PDF Sources</h3>
        <p className="text-xs text-gray-500">No project loaded</p>
      </div>
    );
  }

  if (pdfs.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">PDF Sources</h3>
        <p className="text-xs text-gray-500">No PDFs in this project</p>
      </div>
    );
  }

  const selectedCount = selectedPdfIds.size;
  const totalNodes = pdfs
    .filter(pdf => selectedPdfIds.has(pdf.document_id))
    .reduce((sum, pdf) => sum + pdf.node_count, 0);
  const totalEdges = pdfs
    .filter(pdf => selectedPdfIds.has(pdf.document_id))
    .reduce((sum, pdf) => sum + pdf.edge_count, 0);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">
          PDF Sources
        </h3>
        <div className="text-xs text-gray-500">
          {selectedCount} of {pdfs.length} selected
        </div>
      </div>

      {/* Add PDFs Button */}
      {currentProject && (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={(e) => handleAddPdfs(e.target.files)}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isAdding || isUpdating}
            className="w-full mb-3 px-3 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add More PDFs
          </button>
        </>
      )}

      {isUpdating && (
        <div className="mb-3 flex items-center gap-2 text-xs text-blue-400 bg-blue-900/20 rounded px-2 py-1">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Updating graph...</span>
        </div>
      )}

      {selectedCount > 0 && (
        <div className="mb-3 text-xs text-gray-400 bg-gray-900/50 rounded px-2 py-1.5">
          <div className="flex justify-between">
            <span>Total Nodes:</span>
            <span className="font-mono text-blue-400">{totalNodes}</span>
          </div>
          <div className="flex justify-between">
            <span>Total Edges:</span>
            <span className="font-mono text-green-400">{totalEdges}</span>
          </div>
        </div>
      )}

      <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
        {pdfs.map((pdf) => {
          const isSelected = selectedPdfIds.has(pdf.document_id);
          
          return (
            <div
              key={pdf.document_id}
              className={`
                relative rounded-lg p-3 transition-all
                ${isSelected 
                  ? 'bg-blue-900/30 border border-blue-500/50' 
                  : 'bg-gray-900/50 border border-gray-700'
                }
                ${!pdf.processed ? 'opacity-50' : ''}
                ${isUpdating ? 'opacity-50' : ''}
              `}
            >
              <div className="flex items-start gap-2">
                <button
                  onClick={() => handleTogglePdf(pdf.document_id)}
                  disabled={isUpdating || !pdf.processed}
                  className="flex-1 text-left flex items-start gap-2 min-w-0"
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {!pdf.processed ? (
                      <XCircle className="w-4 h-4 text-red-400" />
                    ) : isSelected ? (
                      <CheckCircle2 className="w-4 h-4 text-blue-400" />
                    ) : (
                      <FileText className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                
                    <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-gray-200 truncate mb-1">
                      {pdf.filename}
                    </div>
                    
                    {pdf.processed ? (
                      <div className="flex gap-3 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                          {pdf.node_count} nodes
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-green-500"></span>
                          {pdf.edge_count} edges
                        </span>
                      </div>
                    ) : (
                      <div className="text-xs text-red-400">
                        Processing failed
                      </div>
                    )}
                    
                    {pdf.processed && Object.keys(pdf.entity_counts).length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {Object.entries(pdf.entity_counts)
                          .sort(([, a], [, b]) => b - a)
                          .slice(0, 3)
                          .map(([type, count]) => (
                            <span 
                              key={type}
                              className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400"
                            >
                              {type.replace('_', ' ')}: {count}
                            </span>
                          ))
                        }
                      </div>
                    )}
                  </div>
                </button>
                
                {/* Delete Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeletePdf(pdf.document_id, pdf.filename);
                  }}
                  disabled={isUpdating}
                  className="flex-shrink-0 p-1.5 hover:bg-red-900/30 rounded transition-colors group disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Delete PDF"
                >
                  <Trash2 className="w-4 h-4 text-gray-500 group-hover:text-red-400" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 pt-3 border-t border-gray-700">
        <button
          onClick={loadProjectPdfs}
          disabled={isUpdating}
          className="w-full text-xs text-gray-400 hover:text-gray-300 transition-colors"
        >
          Refresh PDF List
        </button>
      </div>
    </div>
  );
};

