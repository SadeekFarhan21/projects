import React, { useState } from 'react';
import { googleLogout } from '@react-oauth/google';
import { useAuth } from '../contexts/AuthContext';
import { useStore } from '../store/useStore';
import { User, LogOut, Settings, FolderOpen } from 'lucide-react';

export const UserProfile: React.FC = () => {
  const { user, logout } = useAuth();
  const { setShowProjectSelection } = useStore();
  const [isOpen, setIsOpen] = useState(false);

  const handleLogout = () => {
    googleLogout();
    logout();
  };

  if (!user) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-700 transition-colors duration-200"
      >
        {user.picture ? (
          <img
            src={user.picture}
            alt={user.name}
            className="w-8 h-8 rounded-full object-cover"
          />
        ) : (
          <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
            <User className="w-5 h-5 text-gray-300" />
          </div>
        )}
        <span className="text-white text-sm font-medium hidden sm:block">
          {user.name}
        </span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          
          {/* Dropdown */}
          <div className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-lg shadow-xl border border-gray-700 z-20">
            <div className="p-4 border-b border-gray-700">
              <div className="flex items-center space-x-3">
                {user.picture ? (
                  <img
                    src={user.picture}
                    alt={user.name}
                    className="w-12 h-12 rounded-full object-cover"
                  />
                ) : (
                  <div className="w-12 h-12 rounded-full bg-gray-600 flex items-center justify-center">
                    <User className="w-6 h-6 text-gray-300" />
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{user.name}</p>
                  <p className="text-gray-400 text-sm truncate">{user.email}</p>
                </div>
              </div>
            </div>
            
            <div className="py-2">
              <button
                onClick={() => {
                  setIsOpen(false);
                  setShowProjectSelection(true);
                }}
                className="w-full px-4 py-2 text-left text-gray-300 hover:bg-gray-700 hover:text-white transition-colors duration-200 flex items-center space-x-3"
              >
                <FolderOpen className="w-4 h-4" />
                <span>Switch Project</span>
              </button>
              
              <button
                onClick={() => {
                  setIsOpen(false);
                  // Add settings functionality here if needed
                }}
                className="w-full px-4 py-2 text-left text-gray-300 hover:bg-gray-700 hover:text-white transition-colors duration-200 flex items-center space-x-3"
              >
                <Settings className="w-4 h-4" />
                <span>Settings</span>
              </button>
              
              <button
                onClick={() => {
                  setIsOpen(false);
                  handleLogout();
                }}
                className="w-full px-4 py-2 text-left text-gray-300 hover:bg-gray-700 hover:text-white transition-colors duration-200 flex items-center space-x-3"
              >
                <LogOut className="w-4 h-4" />
                <span>Sign Out</span>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};