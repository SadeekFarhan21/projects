import { useState } from 'react';
import { Download, FileJson, FileSpreadsheet, Image as ImageIcon } from 'lucide-react';
import { useStore } from '@/store/useStore';
import toast from 'react-hot-toast';

export function ExportMenu() {
  const { graphData } = useStore();
  const [isOpen, setIsOpen] = useState(false);

  if (!graphData) return null;

  const exportAsJSON = () => {
    try {
      const dataStr = JSON.stringify(graphData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `empirica-graph-${Date.now()}.json`;
      link.click();
      URL.revokeObjectURL(url);
      toast.success('Graph data exported as JSON');
    } catch (error) {
      toast.error('Failed to export JSON');
    }
  };

  const exportAsCSV = () => {
    try {
      // Export nodes
      const nodesCsv = [
        ['ID', 'Type', 'Connections', 'Occurrences'].join(','),
        ...graphData.nodes.map((node) =>
          [
            `"${node.id}"`,
            node.group,
            node.value,
            node.metadata.count || 0,
          ].join(',')
        ),
      ].join('\n');

      const nodesBlob = new Blob([nodesCsv], { type: 'text/csv' });
      const nodesUrl = URL.createObjectURL(nodesBlob);
      const nodesLink = document.createElement('a');
      nodesLink.href = nodesUrl;
      nodesLink.download = `empirica-nodes-${Date.now()}.csv`;
      nodesLink.click();
      URL.revokeObjectURL(nodesUrl);

      // Export edges
      const edgesCsv = [
        ['Source', 'Target', 'Weight', 'Type', 'Evidence'].join(','),
        ...graphData.edges.map((edge) =>
          [
            `"${edge.source}"`,
            `"${edge.target}"`,
            edge.value,
            edge.metadata?.relationship_type || 'CO_OCCURRENCE',
            `"${edge.title.replace(/"/g, '""')}"`,
          ].join(',')
        ),
      ].join('\n');

      const edgesBlob = new Blob([edgesCsv], { type: 'text/csv' });
      const edgesUrl = URL.createObjectURL(edgesBlob);
      const edgesLink = document.createElement('a');
      edgesLink.href = edgesUrl;
      edgesLink.download = `empirica-edges-${Date.now()}.csv`;
      edgesLink.click();
      URL.revokeObjectURL(edgesUrl);

      toast.success('Graph data exported as CSV files');
    } catch (error) {
      toast.error('Failed to export CSV');
    }
  };

  const exportAsImage = () => {
    try {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;
      if (!canvas) {
        toast.error('Canvas not found');
        return;
      }

      canvas.toBlob((blob) => {
        if (!blob) {
          toast.error('Failed to create image');
          return;
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `empirica-graph-${Date.now()}.png`;
        link.click();
        URL.revokeObjectURL(url);
        toast.success('Graph exported as image');
      });
    } catch (error) {
      toast.error('Failed to export image');
    }
  };

  return (
    <div className="fixed right-4 bottom-4 z-30">
      {isOpen && (
        <div className="mb-2 space-y-2">
          <button
            onClick={exportAsJSON}
            className="flex items-center space-x-2 w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-all shadow-lg"
          >
            <FileJson className="w-5 h-5" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={exportAsCSV}
            className="flex items-center space-x-2 w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-all shadow-lg"
          >
            <FileSpreadsheet className="w-5 h-5" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={exportAsImage}
            className="flex items-center space-x-2 w-full px-4 py-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-all shadow-lg"
          >
            <ImageIcon className="w-5 h-5" />
            <span>Export Image</span>
          </button>
        </div>
      )}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-3 bg-gradient-to-r from-primary-600 to-primary-500 text-white font-semibold rounded-lg shadow-lg hover:from-primary-700 hover:to-primary-600 transition-all flex items-center justify-center space-x-2"
      >
        <Download className="w-5 h-5" />
        <span>Export</span>
      </button>
    </div>
  );
}

