#!/usr/bin/env python3
"""
CTI-NLP System Setup and Installation Script
Automates the setup process for the CTI-NLP system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command(f"{sys.executable} -m venv venv", 
                      "Creating virtual environment"):
        return False
    
    return True

def activate_virtual_environment():
    """Return activation command for the current OS"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        return "source venv/bin/activate"

def install_requirements():
    """Install Python requirements"""
    activate_cmd = activate_virtual_environment()
    
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", 
                      "Installing Python packages"):
        return False
    
    return True

def download_models():
    """Download required ML models"""
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    download_script = '''
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

print("Downloading BERT model for NER...")
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    print("✅ BERT model downloaded successfully")
except Exception as e:
    print(f"❌ Error downloading BERT model: {e}")
'''
    
    with open("download_models.py", "w") as f:
        f.write(download_script)
    
    success = run_command(f"{python_cmd} download_models.py", 
                         "Downloading pre-trained models")
    
    # Clean up
    os.remove("download_models.py")
    return success

def create_directories():
    """Create necessary directories"""
    directories = [
        "models/saved",
        "logs",
        "data/processed",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def run_tests():
    """Run basic tests to verify installation"""
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    test_script = '''
import sys
import importlib

packages = [
    "sklearn", "pandas", "numpy", "transformers", 
    "fastapi", "uvicorn", "matplotlib", "seaborn"
]

print("Testing package imports...")
failed_imports = []

for package in packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"❌ {package}: {e}")
        failed_imports.append(package)

if failed_imports:
    print(f"\\n❌ Failed to import: {', '.join(failed_imports)}")
    sys.exit(1)
else:
    print("\\n✅ All packages imported successfully!")
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    success = run_command(f"{python_cmd} test_imports.py", 
                         "Testing package imports")
    
    # Clean up
    os.remove("test_imports.py")
    return success

def main():
    """Main setup function"""
    print("🚀 CTI-NLP System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        (setup_virtual_environment, "Setting up virtual environment"),
        (install_requirements, "Installing Python packages"),
        (create_directories, "Creating project directories"),
        (download_models, "Downloading pre-trained models"),
        (run_tests, "Running installation tests")
    ]
    
    for step_func, description in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {description}")
            sys.exit(1)
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 CTI-NLP System Setup Complete!")
    print("=" * 50)
    
    activation_cmd = activate_virtual_environment()
    
    print("\n📋 Next Steps:")
    print(f"1. Activate virtual environment: {activation_cmd}")
    print("2. Train models: python train_model.py")
    print("3. Start API server: python api/main.py")
    print("4. Open dashboard: frontend/dashboard.html")
    
    print("\n📚 Available Commands:")
    print("• Train models: python train_model.py")
    print("• Start API: uvicorn api.main:app --reload")
    print("• Run tests: pytest tests/")
    
    print("\n🔗 Useful Links:")
    print("• API Documentation: http://localhost:8000/docs")
    print("• Dashboard: file:///path/to/frontend/dashboard.html")
    print("• Logs: logs/cti_nlp.log")

if __name__ == "__main__":
    main()