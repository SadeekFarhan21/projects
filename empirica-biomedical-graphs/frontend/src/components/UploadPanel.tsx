import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Loader2, FileText, X, Network } from 'lucide-react';
import { apiService } from '@/services/api';
import { useStore } from '@/store/useStore';
import toast from 'react-hot-toast';

export function UploadPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [projectName, setProjectName] = useState('');
  
  const { setProcessingStatus, setIsProcessing, setGraphData, setCurrentProject, setPdfs, setShowUploadPanel } = useStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: true,
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one PDF file');
      return;
    }

    setIsProcessing(true);
    const loadingToast = toast.loading('Uploading PDFs...');

    try {
      // Start processing
      const status = await apiService.uploadPDFs(files, projectName, false);
      setProcessingStatus(status);

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const updatedStatus = await apiService.getStatus(status.job_id);
          setProcessingStatus(updatedStatus);

          if (updatedStatus.status === 'completed') {
            clearInterval(pollInterval);
            setIsProcessing(false);
            
            if (updatedStatus.result) {
              console.log('Graph data received:', updatedStatus.result);
              setGraphData(updatedStatus.result);
              
              // Extract project ID from metadata and load project info
              const projectId = updatedStatus.result.metadata?.project_id;
              if (projectId) {
                try {
                  // Load project info
                  const projects = await apiService.listProjects();
                  const project = projects.find(p => p.project_id === projectId);
                  if (project) {
                    setCurrentProject(project);
                    setPdfs(project.pdfs);
                  }
                } catch (error) {
                  console.error('Failed to load project info:', error);
                }
              }
              
              toast.success('Knowledge graph generated successfully!', {
                id: loadingToast,
              });
              
              // Clear files after successful upload
              setFiles([]);
              setProjectName('');
              
              // Hide upload panel and show project
              setShowUploadPanel(false);
            } else {
              console.error('No result in completed status:', updatedStatus);
              toast.error('No graph data received');
            }
          } else if (updatedStatus.status === 'failed') {
            clearInterval(pollInterval);
            setIsProcessing(false);
            toast.error(`Processing failed: ${updatedStatus.message}`, {
              id: loadingToast,
            });
          } else {
            toast.loading(
              `${updatedStatus.message} (${Math.round(updatedStatus.progress * 100)}%)`,
              { id: loadingToast }
            );
          }
        } catch (error) {
          clearInterval(pollInterval);
          setIsProcessing(false);
          toast.error('Failed to check processing status', { id: loadingToast });
        }
      }, 2000);
    } catch (error: any) {
      setIsProcessing(false);
      toast.error(error.response?.data?.detail || 'Upload failed', {
        id: loadingToast,
      });
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl p-8 border border-gray-700">
        <div className="text-center mb-8">
          <Network className="w-24 h-24 mx-auto mb-4 text-gray-600" />
          <h1 className="text-4xl font-bold text-white mb-2">
            Empirica
          </h1>
          <p className="text-gray-400">
            Transform biomedical PDFs into interactive knowledge graphs
          </p>
        </div>

        {/* Project Name */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Project Name (Optional)
          </label>
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="My Research Project"
            className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            isDragActive
              ? 'border-primary-500 bg-primary-500/10'
              : 'border-gray-600 hover:border-gray-500 bg-gray-800/50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          {isDragActive ? (
            <p className="text-white font-medium">Drop the PDFs here...</p>
          ) : (
            <>
              <p className="text-white font-medium mb-1">
                Click to upload or drag and drop
              </p>
              <p className="text-gray-400 text-sm">
                PDF files only • Multiple files supported
              </p>
            </>
          )}
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="mt-6 space-y-2">
            <p className="text-sm font-medium text-gray-300">
              Selected Files ({files.length})
            </p>
            <div className="max-h-48 overflow-y-auto space-y-2">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between bg-gray-800 rounded-lg p-3"
                >
                  <div className="flex items-center space-x-3 flex-1 min-w-0">
                    <FileText className="w-5 h-5 text-primary-400 flex-shrink-0" />
                    <span className="text-sm text-white truncate">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-400 flex-shrink-0">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="ml-2 p-1 hover:bg-gray-700 rounded"
                  >
                    <X className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={files.length === 0}
          className="w-full mt-8 py-3 px-6 bg-gradient-to-r from-primary-600 to-primary-500 text-white font-semibold rounded-lg shadow-lg hover:from-primary-700 hover:to-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
        >
          {false ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              <span>Generate Knowledge Graph</span>
            </>
          )}
        </button>

        {/* Import Project Option */}
        <div className="mt-4 text-center">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-700"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-gray-900 text-gray-400">or</span>
            </div>
          </div>
          <button
            onClick={() => {
              const input = document.createElement('input');
              input.type = 'file';
              input.accept = '.json';
              input.onchange = async (e: any) => {
                const file = e.target?.files?.[0];
                if (!file) return;
                try {
                  const text = await file.text();
                  const data = JSON.parse(text);
                  console.log('UploadPanel import - raw data:', data);
                  
                  const res = await apiService.importProject(data);
                  console.log('UploadPanel import - API response:', res);
                  
                  if (res.project_id) {
                    // Load the newly imported project
                    const projects = await apiService.listProjects();
                    const importedProject = projects.find(p => p.project_id === res.project_id);
                    
                    if (importedProject) {
                      setCurrentProject(importedProject);
                      setPdfs(importedProject.pdfs);
                      
                      // Load the graph from selected PDFs
                      const graph = await apiService.getProjectGraph(res.project_id, true);
                      setGraphData(graph);
                      
                      toast.success(`Imported project "${res.project_name}" with ${res.pdf_count} PDFs!`);
                    } else {
                      toast.error('Failed to load imported project');
                    }
                  } else {
                    toast.error('Import failed: No project created');
                    console.error('Import response:', res);
                  }
                } catch (error: any) {
                  console.error('Import error:', error);
                  toast.error(`Failed to import: ${error.message || 'Unknown error'}`);
                }
              };
              input.click();
            }}
            className="mt-4 px-6 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-all"
          >
            Import Existing Project
          </button>
        </div>
      </div>
    </div>
  );
}

