#!/bin/bash
# Setup script for Phone Usage Detection System

echo "🚀 Setting up Phone Usage Detection System..."

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Create output directory
mkdir -p src/output

echo "✅ Setup complete!"
echo ""
echo "🎯 Quick start:"
echo "cd src"
echo "python3 main.py ../test_videos/video_1.mp4 --hand-conf 0.1 --phone-conf 0.3"
echo ""
echo "📖 For more details, see README.md"
