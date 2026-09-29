import React from 'react';
import { useGoogleLogin } from '@react-oauth/google';
import { useAuth } from '../contexts/AuthContext';
import { Brain, FileText, Search, BarChart3, Shield, Zap, ArrowRight } from 'lucide-react';

export const LoginComponent: React.FC = () => {
  const { user, isAuthenticated, isLoading } = useAuth();

  const googleLogin = useGoogleLogin({
    onSuccess: (tokenResponse) => {
      const accessToken = tokenResponse.access_token;

      // Fetch user info from Google
      fetch(`https://www.googleapis.com/oauth2/v2/userinfo?access_token=${accessToken}`)
        .then(res => res.json())
        .then(data => {
          const userData = {
            id: data.id,
            name: data.name,
            email: data.email,
            picture: data.picture,
          };

          // Store in localStorage
          localStorage.setItem('auth_token', accessToken);
          localStorage.setItem('user_data', JSON.stringify(userData));

          // Reload the page to update the auth state
          window.location.reload();
        })
        .catch(error => {
          console.error('Error fetching user info:', error);
        });
    },
    onError: (error: any) => {
      console.error('Login failed:', error);
    },
    scope: 'openid email profile',
  });

  const handleLogin = () => {
    googleLogin();
  };


  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
        <div className="text-white text-lg">Loading...</div>
      </div>
    );
  }

  // If user is authenticated, don't show anything - let the main App handle routing
  if (isAuthenticated && user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="relative z-10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Empirica</span>
          </div>
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
            <a href="#about" className="text-gray-300 hover:text-white transition-colors">About</a>
            <a href="#contact" className="text-gray-300 hover:text-white transition-colors">Contact</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column - Content */}
            <div className="space-y-8">
              <div className="space-y-4">
                <h1 className="text-5xl lg:text-6xl font-bold text-white leading-tight">
                  Empirical Research,
                  <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                    {' '}Scaled with Intelligence
                  </span>
                </h1>
                <p className="text-xl text-gray-300 leading-relaxed">
                  Transform your biomedical research with AI-powered document analysis,
                  knowledge graphs, and intelligent insights. Process thousands of PDFs
                  in minutes, not months.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={handleLogin}
                  className="group bg-white hover:bg-gray-100 text-gray-900 font-semibold py-4 px-8 rounded-xl transition-all duration-200 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24">
                    <path
                      fill="#4285F4"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="#34A853"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="#FBBC05"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                      fill="#EA4335"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                  </svg>
                  Get Started with Google
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
                <button className="border-2 border-white/20 hover:border-white/40 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-200 hover:bg-white/5">
                  Learn More
                </button>
              </div>

              <div className="flex items-center space-x-6 text-sm text-gray-400">
                <div className="flex items-center space-x-2">
                  <Shield className="w-4 h-4" />
                  <span>Secure & Private</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4" />
                  <span>Lightning Fast</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4" />
                  <span>AI-Powered</span>
                </div>
              </div>
            </div>

            {/* Right Column - Visual */}
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-3xl blur-3xl"></div>
              <div className="relative bg-gray-800/50 backdrop-blur-sm rounded-3xl p-8 border border-gray-700/50">
                <div className="space-y-6">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span className="text-gray-400 text-sm ml-4">Empirica Dashboard</span>
                  </div>

                  <div className="space-y-4">
                    <div className="bg-gray-700/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <FileText className="w-4 h-4 text-blue-400" />
                        <span className="text-white text-sm font-medium">Research Papers</span>
                      </div>
                      <div className="space-y-2">
                        <div className="h-2 bg-gray-600 rounded-full">
                          <div className="h-2 bg-blue-500 rounded-full w-3/4"></div>
                        </div>
                        <div className="text-xs text-gray-400">Processing 1,247 documents...</div>
                      </div>
                    </div>

                    <div className="bg-gray-700/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Search className="w-4 h-4 text-green-400" />
                        <span className="text-white text-sm font-medium">Knowledge Graph</span>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="bg-green-500/20 rounded p-2 text-center">
                          <div className="text-green-400 text-lg font-bold">2.3k</div>
                          <div className="text-xs text-gray-400">Entities</div>
                        </div>
                        <div className="bg-blue-500/20 rounded p-2 text-center">
                          <div className="text-blue-400 text-lg font-bold">5.7k</div>
                          <div className="text-xs text-gray-400">Relations</div>
                        </div>
                        <div className="bg-purple-500/20 rounded p-2 text-center">
                          <div className="text-purple-400 text-lg font-bold">89%</div>
                          <div className="text-xs text-gray-400">Accuracy</div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-700/50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <BarChart3 className="w-4 h-4 text-purple-400" />
                        <span className="text-white text-sm font-medium">Insights</span>
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Key Findings</span>
                          <span className="text-white">47</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Trends Identified</span>
                          <span className="text-white">12</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Gaps Found</span>
                          <span className="text-white">8</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="px-6 py-20 bg-black/20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Powerful Research Tools</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Everything you need to accelerate your biomedical research and discovery
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-blue-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6">
                <FileText className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Document Processing</h3>
              <p className="text-gray-300 leading-relaxed">
                Upload and process thousands of PDF documents with advanced OCR and text extraction capabilities.
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-green-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-green-500/20 rounded-xl flex items-center justify-center mb-6">
                <Brain className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">AI-Powered Analysis</h3>
              <p className="text-gray-300 leading-relaxed">
                Leverage cutting-edge AI to extract entities, relationships, and insights from your research documents.
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-purple-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center mb-6">
                <Search className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Knowledge Graphs</h3>
              <p className="text-gray-300 leading-relaxed">
                Visualize complex relationships and connections in your research data with interactive knowledge graphs.
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-yellow-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-yellow-500/20 rounded-xl flex items-center justify-center mb-6">
                <BarChart3 className="w-6 h-6 text-yellow-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Analytics & Insights</h3>
              <p className="text-gray-300 leading-relaxed">
                Get comprehensive analytics and actionable insights from your research data with advanced visualization tools.
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-red-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-red-500/20 rounded-xl flex items-center justify-center mb-6">
                <Shield className="w-6 h-6 text-red-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Secure & Private</h3>
              <p className="text-gray-300 leading-relaxed">
                Your data is protected with enterprise-grade security and privacy controls. Research with confidence.
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-indigo-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-indigo-500/20 rounded-xl flex items-center justify-center mb-6">
                <Zap className="w-6 h-6 text-indigo-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Lightning Fast</h3>
              <p className="text-gray-300 leading-relaxed">
                Process and analyze your research documents at unprecedented speed with our optimized infrastructure.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-12 border-t border-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="text-center text-gray-400">
            <p className="mb-4">By signing in, you agree to our Terms of Service and Privacy Policy</p>
            <p>&copy; 2024 Empirica. All rights reserved.</p>
          </div>
        </div>
      </div>
    </div>
  );
};