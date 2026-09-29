# Empirica - Frontend

Interactive React application for visualizing biomedical knowledge graphs.

## Features

- 🎨 **2D/3D Graph Visualization** - Toggle between 2D and 3D force-directed graphs
- 🔍 **Interactive Exploration** - Click nodes to explore connections and relationships
- 🎯 **Smart Filtering** - Filter by entity type, connection count, or search
- 📊 **Analytics Dashboard** - View graph statistics and community detection
- 💫 **Beautiful UI** - Modern, responsive design with Tailwind CSS
- ⚡ **Real-time Updates** - Live processing status with progress tracking

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (default: http://localhost:8000)
```

### 3. Run Development Server

```bash
npm run dev
```

Frontend runs on `http://localhost:5173`

### 4. Build for Production

```bash
npm run build
npm run preview  # Preview production build
```

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **react-force-graph-2d/3d** - Graph visualization
- **Recharts** - Analytics charts
- **Zustand** - State management
- **Axios** - API client
- **react-dropzone** - File uploads
- **react-hot-toast** - Notifications
- **Lucide React** - Icons

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ForceGraph2DView.tsx    # 2D graph visualization
│   │   ├── ForceGraph3DView.tsx    # 3D graph visualization
│   │   ├── UploadPanel.tsx         # PDF upload interface
│   │   ├── Sidebar.tsx             # Filters and controls
│   │   ├── NodeDetails.tsx         # Selected node info
│   │   ├── Analytics.tsx           # Analytics dashboard
│   │   └── ProcessingOverlay.tsx   # Loading state
│   ├── services/
│   │   └── api.ts                  # API client
│   ├── store/
│   │   └── useStore.ts             # Global state management
│   ├── types/
│   │   └── index.ts                # TypeScript types
│   ├── App.tsx                     # Main app component
│   ├── main.tsx                    # Entry point
│   └── index.css                   # Global styles
├── public/
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```

## Features in Detail

### Graph Visualization

- **Force-Directed Layout**: Physics-based simulation for organic clustering
- **Node Coloring**: Color-coded by entity type (genes, chemicals, diseases, etc.)
- **Node Sizing**: Size indicates number of connections (degree)
- **Edge Tooltips**: Hover to see evidence sentences from papers
- **Focus Mode**: Click a node to highlight its neighborhood

### Filtering & Search

- **Entity Type Filter**: Show/hide specific entity types
- **Minimum Degree**: Filter out low-connection nodes
- **Search**: Find specific entities by name
- **Top-N Filter**: Show only the most connected nodes

### Analytics

- **Entity Distribution**: Pie chart of entity types
- **Centrality Analysis**: Most important nodes by betweenness centrality
- **Community Detection**: Louvain algorithm for cluster identification
- **Graph Metrics**: Density, average degree, node/edge counts

## Usage

1. **Upload PDFs**: Drag & drop or click to upload biomedical PDF files
2. **Wait for Processing**: Monitor progress in real-time
3. **Explore Graph**: Pan, zoom, and click nodes to investigate
4. **Apply Filters**: Use sidebar to focus on specific entity types or search
5. **Toggle View**: Switch between 2D and 3D visualization
6. **View Analytics**: Click "View Analytics" for statistical insights

## Development

```bash
# Install dependencies
npm install

# Run dev server with hot reload
npm run dev

# Type checking
npm run build

# Lint code
npm run lint
```

## Performance Tips

- For large graphs (>500 nodes), use filters to improve performance
- Enable "Show labels" only when focused on specific areas
- Use 2D view for better performance with very large graphs
- Search function helps quickly locate specific entities

