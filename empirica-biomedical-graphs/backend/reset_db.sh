#!/bin/bash
# Reset database and clear all uploaded files

echo "🗑️  Resetting Empirica Database..."
echo ""

# Remove database
if [ -f "synapse_mapper.db" ]; then
    rm synapse_mapper.db
    echo "✅ Database deleted"
else
    echo "ℹ️  No database found"
fi

# Clear uploads folder
if [ -d "uploads" ]; then
    find uploads -type f \( -name "*.pdf" -o -name "*.pkl" \) -delete
    echo "✅ Uploads and RAG indices cleared"
else
    echo "ℹ️  No uploads folder found"
fi

echo ""
echo "🎉 Database reset complete!"
echo ""
echo "Next steps:"
echo "  1. Restart backend: uvicorn app.main:app --reload"
echo "  2. Refresh frontend in browser"
echo ""

