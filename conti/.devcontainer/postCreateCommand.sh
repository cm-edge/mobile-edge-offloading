#!/bin/bash

# Install Python dependencies from all requirements files
echo "Installing Python dependencies..."
pip3 install --user -r cloud/requirements.txt
pip3 install --user -r device/requirements.txt 
pip3 install --user -r edge/requirements.txt

# Install development tools
echo "Installing development tools..."
pip3 install --user pytest pytest-cov autopep8 pylint

# Install mosquitto client for testing MQTT connections
echo "Installing MQTT client tools..."
sudo apt-get update && sudo apt-get install -y mosquitto-clients

# Set up any environment variables needed
echo "Setting up environment variables..."
echo "export PYTHONPATH=/workspaces/conti:$PYTHONPATH" >> ~/.bashrc
echo "export PYTHONPATH=/workspaces/conti:$PYTHONPATH" >> ~/.profile

# Create .env file for VS Code Python extension
echo "Creating .env file for VS Code..."
echo "PYTHONPATH=/workspaces/conti" > /workspaces/conti/.env

# Apply PYTHONPATH to current shell
export PYTHONPATH=/workspaces/conti:$PYTHONPATH

echo "Setup complete!"