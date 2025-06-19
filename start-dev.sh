#!/bin/bash

# Development startup script for CoT Clustering Research Platform

echo "🧠 Starting CoT Clustering Research Platform..."
echo ""

# Check if Python backend dependencies are installed
if [ ! -d "backend/venv" ] && [ ! -f "backend/.python_deps_installed" ]; then
    echo "📦 Installing Python backend dependencies..."
    cd backend
    pip3 install -r requirements.txt
    touch .python_deps_installed
    cd ..
    echo "✅ Python dependencies installed"
    echo ""
fi

# Start Python backend in background
echo "🐍 Starting Python FastAPI backend (port 8000)..."
cd backend
python3 main.py &
PYTHON_PID=$!
cd ..

# Wait a moment for Python backend to start
sleep 3

# Start Next.js frontend
echo "⚛️  Starting Next.js frontend (port 3001)..."
npm run dev &
NEXTJS_PID=$!

echo ""
echo "🚀 Both services started!"
echo ""
echo "📱 Frontend: http://localhost:3001"
echo "🔧 Backend API: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait

# Cleanup
echo ""
echo "🛑 Stopping services..."
kill $PYTHON_PID 2>/dev/null
kill $NEXTJS_PID 2>/dev/null
echo "✅ Services stopped" 