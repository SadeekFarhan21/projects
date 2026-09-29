import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { apiService } from '../services/api';
import { ProjectInfo } from '../types';
import { 
  Plus, 
  FolderOpen, 
  Calendar, 
  FileText, 
  Trash2, 
  Edit3, 
  Loader2,
  AlertCircle 
} from 'lucide-react';

interface ProjectSelectionProps {
  onProjectSelect: (project: ProjectInfo) => void;
  onCreateNew: () => void;
  onProjectDeleted?: () => void;
}

export const ProjectSelection: React.FC<ProjectSelectionProps> = ({
  onProjectSelect,
  onCreateNew,
  onProjectDeleted
}) => {
  const { user } = useAuth();
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingProject, setEditingProject] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState('');

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      setError(null);
      const projectList = await apiService.listProjects();
      setProjects(projectList);
    } catch (err) {
      console.error('Error loading projects:', err);
      setError('Failed to load projects. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteProject = async (projectId: string, projectName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${projectName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      setLoading(true);
      await apiService.deleteProject(projectId);
      
      // Reload projects list
      await loadProjects();
      
      // Notify parent component that a project was deleted
      if (onProjectDeleted) {
        onProjectDeleted();
      }
      
    } catch (err) {
      console.error('Error deleting project:', err);
      setError('Failed to delete project');
    } finally {
      setLoading(false);
    }
  };

  const handleRenameProject = async (projectId: string, newName: string) => {
    if (!newName.trim()) return;

    try {
      setLoading(true);
      await apiService.renameProject(projectId, newName.trim());
      
      // Reload projects list
      await loadProjects();
      
    } catch (err) {
      console.error('Error renaming project:', err);
      setError('Failed to rename project');
    } finally {
      setLoading(false);
      setEditingProject(null);
      setNewProjectName('');
    }
  };

  const startEditing = (project: ProjectInfo) => {
    setEditingProject(project.project_id);
    setNewProjectName(project.name);
  };

  const cancelEditing = () => {
    setEditingProject(null);
    setNewProjectName('');
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="text-center text-white">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-lg">Loading your projects...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Welcome back, {user?.name?.split(' ')[0]}!
          </h1>
          <p className="text-gray-300 text-lg">
            Select a project to continue or create a new one
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4 mb-6 flex items-center">
            <AlertCircle className="w-5 h-5 text-red-400 mr-3" />
            <span className="text-red-300">{error}</span>
          </div>
        )}

        {/* Projects Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {/* Create New Project Card */}
          <div
            onClick={onCreateNew}
            className="bg-gray-800/50 border-2 border-dashed border-gray-600 hover:border-blue-500 rounded-lg p-6 cursor-pointer transition-all duration-200 hover:bg-gray-800/70 group"
          >
            <div className="text-center">
              <Plus className="w-12 h-12 text-gray-400 group-hover:text-blue-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Create New Project</h3>
              <p className="text-gray-400 group-hover:text-gray-300">
                Start a new research project
              </p>
            </div>
          </div>

          {/* Existing Projects */}
          {projects.map((project) => (
            <div
              key={project.project_id}
              className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 hover:bg-gray-800/70 transition-all duration-200 group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <FolderOpen className="w-6 h-6 text-blue-400 mr-3" />
                  <div>
                    {editingProject === project.project_id ? (
                      <input
                        type="text"
                        value={newProjectName}
                        onChange={(e) => setNewProjectName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleRenameProject(project.project_id, newProjectName);
                          } else if (e.key === 'Escape') {
                            cancelEditing();
                          }
                        }}
                        onBlur={() => handleRenameProject(project.project_id, newProjectName)}
                        className="bg-gray-700 text-white px-2 py-1 rounded text-lg font-semibold"
                        autoFocus
                      />
                    ) : (
                      <h3 className="text-lg font-semibold text-white group-hover:text-blue-400">
                        {project.name}
                      </h3>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startEditing(project);
                    }}
                    className="p-1 text-gray-400 hover:text-yellow-400 transition-colors"
                    title="Rename project"
                  >
                    <Edit3 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteProject(project.project_id, project.name);
                    }}
                    className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                    title="Delete project"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-3 mb-4">
                <div className="flex items-center text-sm text-gray-400">
                  <Calendar className="w-4 h-4 mr-2" />
                  <span>Created {formatDate(project.created_at)}</span>
                </div>
                <div className="flex items-center text-sm text-gray-400">
                  <FileText className="w-4 h-4 mr-2" />
                  <span>{project.pdf_count} PDF{project.pdf_count !== 1 ? 's' : ''}</span>
                </div>
              </div>

              {project.description && (
                <p className="text-gray-300 text-sm mb-4 line-clamp-2">
                  {project.description}
                </p>
              )}

              <button
                onClick={() => onProjectSelect(project)}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
              >
                Open Project
              </button>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {projects.length === 0 && (
          <div className="text-center py-12">
            <FolderOpen className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No projects yet</h3>
            <p className="text-gray-400 mb-6">
              Create your first project to start analyzing biomedical PDFs
            </p>
            <button
              onClick={onCreateNew}
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200"
            >
              Create Your First Project
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
