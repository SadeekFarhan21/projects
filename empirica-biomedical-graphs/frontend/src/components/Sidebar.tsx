import { useState } from 'react';
import {
  Search,
  Filter,
  Settings,
  ChevronLeft,
  ChevronRight,
  Box,
  BoxSelect,
  Plus,
  LogOut,
  Download,
  Upload,
} from 'lucide-react';
import { useStore } from '@/store/useStore';
import { ENTITY_COLORS, ENTITY_LABELS, EntityType } from '@/types';
import { apiService } from '@/services/api';
import { PDFSelector } from './PDFSelector';
import { useAuth } from '@/contexts/AuthContext';
import toast from 'react-hot-toast';

export function Sidebar() {
  const {
    sidebarOpen,
    toggleSidebar,
    filterOptions,
    setFilterOptions,
    viewMode,
    setViewMode,
    filteredGraphData,
    graphData,
    currentProject,
    setShowUploadPanel,
    setShowProjectSelection,
  } = useStore();
  
  const { user, logout } = useAuth();

  const [searchQuery, setSearchQuery] = useState('');

  const entityTypes = Object.keys(ENTITY_COLORS) as EntityType[];

  const handleEntityTypeToggle = (type: EntityType) => {
    const current = filterOptions.entityTypes;
    const updated = current.includes(type)
      ? current.filter((t) => t !== type)
      : [...current, type];
    setFilterOptions({ entityTypes: updated });
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setFilterOptions({ searchQuery: query });
  };

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={toggleSidebar}
        className="fixed left-4 top-4 z-30 p-2 bg-gray-900/90 backdrop-blur-sm text-white rounded-lg hover:bg-gray-800 transition-all border border-gray-700"
      >
        {sidebarOpen ? (
          <ChevronLeft className="w-5 h-5" />
        ) : (
          <ChevronRight className="w-5 h-5" />
        )}
      </button>

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full bg-gradient-to-b from-gray-900 to-gray-800 border-r border-gray-700 transition-transform duration-300 z-20 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } w-96 overflow-y-auto`}
      >
        <div className="p-6 pt-20">
          {/* User Profile Card - Clickable to switch projects */}
          {user && (
            <button
              onClick={() => setShowProjectSelection(true)}
              className="mb-6 w-full p-3 bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-lg border border-blue-700/30 backdrop-blur-sm hover:from-blue-900/60 hover:to-purple-900/60 hover:border-blue-600/50 transition-all text-left"
              title="Click to switch projects"
            >
              <div className="flex items-center space-x-3">
                {user.picture ? (
                  <img 
                    src={user.picture} 
                    alt={user.name}
                    className="w-10 h-10 rounded-full border 2px border-blue-500"
                  />
                ) : (
                  <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-white truncate">{user.name}</p>
                  <p className="text-xs text-gray-300 truncate">{user.email}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    logout();
                  }}
                  title="Logout"
                  className="text-gray-400 hover:text-red-400 transition-colors p-1"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </button>
          )}
          
          {/* Header */}
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-white mb-1">
              Empirica
            </h2>
            <p className="text-sm text-gray-400">
              {filteredGraphData
                ? `${filteredGraphData.nodes.length} nodes, ${filteredGraphData.edges.length} edges`
                : 'No data loaded'}
            </p>
          </div>

          {/* New Project Button */}
          <div className="mb-6">
            <button
              onClick={() => setShowUploadPanel(true)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Project</span>
            </button>
          </div>

          {/* PDF Selector */}
          <div className="mb-6">
            <PDFSelector />
          </div>

          {/* Search */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <Search className="w-4 h-4 inline mr-2" />
              Search Entities
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="Search by name..."
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          {/* View Mode */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              <Settings className="w-4 h-4 inline mr-2" />
              View Settings
            </label>
            
            {/* 2D/3D Toggle */}
            <div className="flex space-x-2 mb-3">
              <button
                onClick={() => setViewMode({ dimension: '2d' })}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                  viewMode.dimension === '2d'
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <BoxSelect className="w-4 h-4 inline mr-2" />
                2D
              </button>
              <button
                onClick={() => setViewMode({ dimension: '3d' })}
                className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                  viewMode.dimension === '3d'
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <Box className="w-4 h-4 inline mr-2" />
                3D
              </button>
            </div>

            {/* Show Labels */}
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={viewMode.showLabels}
                onChange={(e) => setViewMode({ showLabels: e.target.checked })}
                className="w-4 h-4 text-primary-600 bg-gray-800 border-gray-600 rounded focus:ring-primary-500"
              />
              <span className="text-sm text-gray-300">Show node labels</span>
            </label>
          </div>

          {/* Filters */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              <Filter className="w-4 h-4 inline mr-2" />
              Filter by Entity Type
            </label>
            <div className="space-y-2">
              {entityTypes.map((type) => (
                <label
                  key={type}
                  className="flex items-center space-x-3 cursor-pointer p-2 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={
                      filterOptions.entityTypes.length === 0 ||
                      filterOptions.entityTypes.includes(type)
                    }
                    onChange={() => handleEntityTypeToggle(type)}
                    className="w-4 h-4 rounded border-gray-600 text-primary-600 focus:ring-primary-500"
                  />
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: ENTITY_COLORS[type] }}
                  />
                  <span className="text-sm text-gray-300 flex-1">
                    {ENTITY_LABELS[type]}
                  </span>
                  <span className="text-xs text-gray-500">
                    {graphData?.nodes.filter((n) => n.group === type).length || 0}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Min Degree Filter */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Minimum Connections: {filterOptions.minDegree}
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={filterOptions.minDegree}
              onChange={(e) =>
                setFilterOptions({ minDegree: parseInt(e.target.value) })
              }
              className="w-full"
            />
          </div>

          {/* Export/Import */}
          <div className="mt-6 pt-6 border-t border-gray-700">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Project Management
            </label>
            <div className="space-y-2">
              <button
                onClick={async () => {
                  if (!currentProject) {
                    toast.error('No project loaded');
                    return;
                  }
                  
                  try {
                    const projectData = await apiService.exportProject(currentProject.project_id);
                    const blob = new Blob([JSON.stringify(projectData, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${currentProject.name}_${Date.now()}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    toast.success('Project exported successfully!');
                  } catch (error) {
                    console.error('Export failed:', error);
                    toast.error('Export failed');
                  }
                }}
                disabled={!currentProject}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:from-gray-700 disabled:to-gray-600 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-all shadow-lg disabled:shadow-none"
              >
                <Download className="w-4 h-4" />
                <span>Export Project</span>
              </button>
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
                      console.log('Importing project:', data);
                      
                      const res = await apiService.importProject(data);
                      console.log('Import response:', res);
                      
                      if (res.project_id) {
                        // Load the newly imported project
                        const projects = await apiService.listProjects();
                        const importedProject = projects.find(p => p.project_id === res.project_id);
                        
                        if (importedProject) {
                          useStore.getState().setCurrentProject(importedProject);
                          useStore.getState().setPdfs(importedProject.pdfs);
                          
                          // Load the graph
                          const graph = await apiService.getProjectGraph(res.project_id, true);
                          useStore.getState().setGraphData(graph);
                          
                          toast.success(`Imported project with ${res.pdf_count} PDFs!`);
                        }
                      } else {
                        toast.error('Import failed: No project created');
                      }
                    } catch (error: any) {
                      console.error('Import error:', error);
                      toast.error(`Import error: ${error.message}`);
                    }
                  };
                  input.click();
                }}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 bg-gradient-to-r from-green-600 to-green-500 hover:from-green-500 hover:to-green-400 rounded-lg text-white font-medium transition-all shadow-lg"
              >
                <Upload className="w-4 h-4" />
                <span>Import Project</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

