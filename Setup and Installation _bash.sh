#!/bin/bash
# setup.sh

echo "Setting up Traffic Monitoring System..."
echo "======================================"

# Create directory structure
echo "Creating directory structure..."
mkdir -p config data/{images/{train,val,test},videos/{input,processed},labels} models logs scripts src/{detection,tracking,gui,utils} tests

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download pre-trained model
echo "Downloading pre-trained YOLOv8 model..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# Create configuration files
echo "Creating configuration files..."
cat > config/classes.txt << EOF
car
bus
truck
motorcycle
van
pickup
trailer
ambulance
traveller
EOF

echo "Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Place your images in data/images/"
echo "2. Run: python scripts/split_dataset.py"
echo "3. Run: python scripts/train_model.py"
echo "4. Run: python main.py --video your_video.mp4"