import { X, TrendingUp, Network, Layers, Target } from 'lucide-react';
import { useStore } from '@/store/useStore';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { ENTITY_COLORS, ENTITY_LABELS } from '@/types';

export function Analytics() {
  const { analyticsOpen, toggleAnalytics, graphData } = useStore();

  if (!analyticsOpen || !graphData) return null;

  const analytics = graphData.metadata.analytics;

  // Prepare data for charts
  const entityData = analytics?.entity_counts
    ? Object.entries(analytics.entity_counts).map(([type, count]) => ({
        name: ENTITY_LABELS[type as keyof typeof ENTITY_LABELS] || type,
        value: count,
        color: ENTITY_COLORS[type as keyof typeof ENTITY_COLORS] || '#6b7280',
      }))
    : [];

  const centralityData = analytics?.centrality_scores
    ? Object.entries(analytics.centrality_scores)
        .slice(0, 10)
        .map(([node, score]) => ({
          name: node.length > 20 ? node.substring(0, 20) + '...' : node,
          score: Math.round(score * 1000) / 1000,
        }))
    : [];

  const communityData = analytics?.communities
    ? analytics.communities.map((community, idx) => ({
        name: `Community ${idx + 1}`,
        size: community.length,
      }))
    : [];

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-40 overflow-y-auto">
      <div className="min-h-screen p-8 flex items-center justify-center">
        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl shadow-2xl max-w-6xl w-full border border-gray-700">
          {/* Header */}
          <div className="p-6 border-b border-gray-700 flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold text-white mb-1">
                Graph Analytics
              </h2>
              <p className="text-gray-400">
                Statistical analysis of the knowledge graph
              </p>
            </div>
            <button
              onClick={toggleAnalytics}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <X className="w-6 h-6 text-gray-400" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {/* Overview Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <StatCard
                icon={<Network className="w-6 h-6" />}
                label="Total Nodes"
                value={graphData.nodes.length}
                color="text-blue-400"
              />
              <StatCard
                icon={<TrendingUp className="w-6 h-6" />}
                label="Total Edges"
                value={graphData.edges.length}
                color="text-green-400"
              />
              <StatCard
                icon={<Target className="w-6 h-6" />}
                label="Density"
                value={(analytics?.density || 0).toFixed(4)}
                color="text-purple-400"
              />
              <StatCard
                icon={<Layers className="w-6 h-6" />}
                label="Avg Degree"
                value={(analytics?.avg_degree || 0).toFixed(2)}
                color="text-orange-400"
              />
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Entity Distribution */}
              <ChartCard title="Entity Type Distribution">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={entityData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry) => `${entry.name}: ${entry.value}`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {entityData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* Top Central Nodes */}
              <ChartCard title="Top 10 Most Central Nodes">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={centralityData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis
                      dataKey="name"
                      type="category"
                      width={150}
                      stroke="#9ca3af"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Bar dataKey="score" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* Community Sizes */}
              {communityData.length > 0 && (
                <ChartCard title="Community Sizes">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={communityData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1f2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                      <Bar dataKey="size" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

              {/* Communities List */}
              <ChartCard title="Detected Communities">
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {analytics?.communities?.map((community, idx) => (
                    <div
                      key={idx}
                      className="bg-gray-800 rounded-lg p-3"
                    >
                      <p className="text-sm font-semibold text-white mb-1">
                        Community {idx + 1} ({community.length} nodes)
                      </p>
                      <p className="text-xs text-gray-400">
                        {community.slice(0, 5).join(', ')}
                        {community.length > 5 && '...'}
                      </p>
                    </div>
                  ))}
                </div>
              </ChartCard>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
}) {
  return (
    <div className="bg-gray-800/50 rounded-xl p-6">
      <div className={`${color} mb-2`}>{icon}</div>
      <p className="text-sm text-gray-400 mb-1">{label}</p>
      <p className="text-3xl font-bold text-white">{value}</p>
    </div>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-800/50 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      {children}
    </div>
  );
}

