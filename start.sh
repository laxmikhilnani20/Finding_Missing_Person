#!/bin/bash

# CCTV Missing Person Detection System - Quick Start Script

echo "🔍 CCTV-Based Missing Person Detection System"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed!"
    echo "Please install docker-compose"
    exit 1
fi

echo "✅ docker-compose found"
echo ""

# Build and run
echo "🏗️  Building Docker image (this may take a few minutes on first run)..."
docker-compose build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting application..."
    echo ""
    echo "================================================"
    echo "   Access the dashboard at:"
    echo "   👉  http://localhost:8501"
    echo "================================================"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo ""
    
    docker-compose up
else
    echo "❌ Build failed! Please check the error messages above."
    exit 1
fi
