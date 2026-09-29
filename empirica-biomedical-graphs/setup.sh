#!/bin/bash

# Empirica Setup Script
# Automated setup for backend and frontend

set -e

echo "🧬 Empirica Setup"
echo "======================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Check Node version
echo "Checking Node.js version..."
node_version=$(node --version)
echo "✓ Found Node.js $node_version"

echo ""
echo "📦 Setting up Backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download scispaCy model
echo "Downloading scispaCy model (this may take a few minutes)..."
echo "Note: This downloads ~1.5GB model from S3..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Database
DATABASE_URL=sqlite:///./synapse_mapper.db

# LLM API Keys (Optional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# LAVA API Configuration
LAVA_SECRET_KEY=aks_live_...
LAVA_CONNECTION_SECRET=cons_live_...
LAVA_PRODUCT_SECRET=ps_live_...
ENABLE_LAVA=true

# Processing Configuration
MAX_UPLOAD_SIZE_MB=100
MAX_CONCURRENT_PROCESSING=4
ENABLE_LLM_EXTRACTION=false

# CORS
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]
EOF
fi

echo "✓ Backend setup complete!"

cd ..
echo ""
echo "🎨 Setting up Frontend..."
cd frontend

# Install dependencies
echo "Installing Node dependencies..."
npm install

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
EOF
fi

echo "✓ Frontend setup complete!"

cd ..
echo ""
echo "✅ Setup Complete!"
echo ""
echo "To start the application:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser!"
echo ""

