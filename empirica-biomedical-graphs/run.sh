#!/bin/bash

# Empirica Run Script
# Starts both backend and frontend in tmux

set -e

echo "🧬 Starting Empirica..."
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Please install it or run backend and frontend manually."
    echo ""
    echo "Manual start:"
    echo "Terminal 1: cd backend && source venv/bin/activate && python -m app.main"
    echo "Terminal 2: cd frontend && npm run dev"
    exit 1
fi

# Create new tmux session
SESSION="empirica"

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# Create new session with backend
tmux new-session -d -s $SESSION -n backend "cd backend && source venv/bin/activate && python -m app.main"

# Create new window for frontend
tmux new-window -t $SESSION -n frontend "cd frontend && npm run dev"

# Attach to session
echo "✅ Started Empirica in tmux session '$SESSION'"
echo ""
echo "To view:"
echo "  tmux attach -t $SESSION"
echo ""
echo "To detach: Ctrl+B then D"
echo "To switch windows: Ctrl+B then N (next) or P (previous)"
echo "To stop: tmux kill-session -t $SESSION"
echo ""
echo "Opening in 3 seconds..."
sleep 3

tmux attach -t $SESSION

