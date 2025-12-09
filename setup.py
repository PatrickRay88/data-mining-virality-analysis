#!/usr/bin/env python3
"""
Setup script for the virality analysis project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"  {e.stderr}")
        return None


def main():
    """Set up the project environment."""
    print("Setting up Virality Analysis Project...")
    
    # Install Python dependencies
    run_command("pip install -r requirements.txt", "Installing Python packages")
    
    # Download spaCy model
    run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Created data directory: {data_dir.absolute()}")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print(f"✓ Created .env file from template")
        print(f"  Please edit .env with your Reddit API credentials")
    elif not env_file.exists():
        print(f"⚠ No .env file found. Create one with your Reddit API credentials")
    
    # Set up Jupyter kernel
    run_command("python -m ipykernel install --user --name virality-analysis", 
                "Setting up Jupyter kernel")
    
    print(f"\n🎉 Setup completed!")
    print(f"\nNext steps:")
    print(f"1. Edit .env file with your Reddit API credentials")
    print(f"2. Start collecting data: python -m src.cli.collect_reddit --subreddit technology")
    print(f"3. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    

if __name__ == "__main__":
    main()