# Clinova - Virtual Clinical Trial Platform

A full-stack application for conducting virtual clinical trials using AI agents to validate medical research hypotheses.

## Project Structure

```
Clinova/
├── frontend/          # React + Vite web application
├── backend/           # FastAPI backend + Python AI agents
├── visualization/     # 3D WebGL visualization (Next.js)
├── docker-compose.yml # Container orchestration
└── README.md          # This file
```

## Tech Stack

### Frontend (`/frontend`)
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5
- **UI Components**: shadcn/ui + Radix UI
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query
- **3D Graphics**: Three.js + React Three Fiber
- **Backend Integration**: REST API + Supabase
- **Routing**: React Router v6
- **Form Handling**: React Hook Form + Zod validation

### Backend (`/backend`)
- **API Framework**: FastAPI
- **Language**: Python 3.11+
- **AI Models**: OpenAI GPT-4, Google Gemini
- **Key Features**:
  - RESTful API for trial management
  - Multi-agent pipeline for clinical trial simulation
  - Clinical trial replication (PREDICT, Barrett, CARE, VALOR, NEPHRIC)
  - Contrast-induced nephropathy (CIN) analysis
  - OMOP CDM data lookup
  - Automated trial validation

### Visualization (`/visualization`)
- **Framework**: Next.js (React)
- **3D Features**: React Three Fiber + Yoga flexbox layouts in WebGL
- **Purpose**: Advanced 3D data visualization

## Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd Clinova
```

2. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_openai_key
# GEMINI_API_KEY=your_gemini_key
```

3. **Start all services**
```bash
docker-compose up
```

This will start:
- Backend API at http://localhost:8000
- Frontend at http://localhost:5173

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# Start the server
python server.py
# Or: uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Backend API will be available at http://localhost:8000

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env
# Edit .env if needed

# Start development server
npm run dev
```

Frontend will be available at http://localhost:5173

## Environment Variables

### Root `.env` (for Docker Compose)
```env
# Backend API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Frontend
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_PROJECT_ID=your_supabase_project_id
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_key
```

### Backend `/backend/.env`
```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Frontend `/frontend/.env`
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_PROJECT_ID=your_project_id
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_key
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

```
POST   /api/trials              # Create new trial
GET    /api/trials              # List all trials
GET    /api/trials/{id}/status  # Get trial status
GET    /api/trials/{id}/results # Get trial results
DELETE /api/trials/{id}         # Delete trial
```

## Usage Example

### Running a Virtual Clinical Trial

1. **Start the backend and frontend** (using docker-compose or manually)

2. **Create a trial via API**:
```bash
curl -X POST http://localhost:8000/api/trials \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do iodinated contrast agents affect eGFR subgroups and risk of AKI in the next 30 days?"
  }'
```

3. **Check trial status**:
```bash
curl http://localhost:8000/api/trials/{run_id}/status
```

4. **Get results** (once completed):
```bash
curl http://localhost:8000/api/trials/{run_id}/results
```

### Using the Frontend

Navigate to http://localhost:5173 and use the web interface to:
- Submit clinical trial questions
- Monitor trial progress
- View results and generated code
- Explore OMOP concept mappings

## Agent Pipeline

The backend runs a sophisticated multi-agent pipeline:

1. **Question Agent** (GPT-4): Transforms natural language to causal PICO format
2. **Design Agent** (GPT-4): Generates Target Trial Emulation (TTE) protocol
3. **Validator Agent** (GPT-4): Validates design against clinical rubric (up to 3 iterations)
4. **OMOP Lookup Agent**: Resolves medical terms to standardized concept IDs
5. **Code Agent** (Gemini): Generates Python implementation code

## Clinical Trials Supported

The system can replicate 5 major randomized controlled trials:

1. **PREDICT (2008)**: CT with IV contrast in CKD + diabetes patients
2. **Barrett et al. (2006)**: CE-MDCT in CKD patients
3. **CARE (2007)**: Cardiac angiography/PCI in CKD ± diabetes
4. **VALOR (2008)**: Coronary angiography in CKD
5. **NEPHRIC (2003)**: High-risk diabetes + CKD angiography

## Development

### Frontend Development
```bash
cd frontend
npm run dev          # Start dev server
npm run build        # Production build
npm run test         # Run tests
npm run lint         # Lint code
```

### Backend Development
```bash
cd backend
python -m agent --question "your question here"  # Run CLI
python server.py     # Run API server
```

### Testing the API
```bash
# Health check
curl http://localhost:8000/

# Create a trial
curl -X POST http://localhost:8000/api/trials \
  -H "Content-Type: application/json" \
  -d '{"question": "How does contrast affect kidney function?"}'
```

## Project Features

- **Virtual Clinical Trials**: Simulate and validate clinical trials using AI
- **Multi-Agent System**: Specialized agents for different pipeline stages
- **Interactive 3D Visualization**: Advanced WebGL-based data visualization
- **Real-time Trial Monitoring**: Track trial progress through web interface
- **OMOP CDM Integration**: Standardized medical terminology lookup
- **Agent-Based Validation**: AI agents validate trial hypotheses autonomously
- **Responsive UI**: Modern, accessible interface built with shadcn/ui
- **RESTful API**: Complete backend API with OpenAPI documentation

## Production Deployment

### Docker Production Build

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production services
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Configuration

For production, make sure to:
1. Use strong, unique API keys
2. Set appropriate CORS origins in backend
3. Use production Supabase instance
4. Configure proper logging
5. Set up monitoring and alerting

## Troubleshooting

### Backend won't start
- Check that OPENAI_API_KEY and GEMINI_API_KEY are set
- Ensure Python 3.11+ is installed
- Verify all dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check VITE_API_URL in frontend/.env
- Verify CORS settings in backend/server.py

### Docker issues
- Ensure Docker and Docker Compose are installed
- Check .env file exists and has correct values
- Try: `docker-compose down -v && docker-compose up --build`

## License

See LICENSE file.

## Contributing

This is a research project focused on virtual clinical trial validation using AI agents. Contributions are welcome!

## Acknowledgments

Built for advancing medical research through AI-powered virtual clinical trials.
