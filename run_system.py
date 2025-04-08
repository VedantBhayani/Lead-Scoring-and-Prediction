import os
import subprocess
import sys
import time
import threading
import webbrowser

def run_backend():
    """Run the backend FastAPI server"""
    os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "MAIN")
    backend_script = os.path.join(os.getcwd(), "MAIN", "src", "backend", "api.py")
    
    try:
        print("\n====== Starting Backend API Server ======\n")
        backend_process = subprocess.Popen(
            [sys.executable, backend_script],
            # Redirect stdout and stderr to current terminal
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Print output in real-time
        for line in backend_process.stdout:
            print(f"[BACKEND] {line.strip()}")
            
        return backend_process
    except Exception as e:
        print(f"Error starting backend: {str(e)}")
        return None

def run_frontend():
    """Run the frontend development server"""
    frontend_dir = os.path.join(os.getcwd(), "Frontend")
    
    # Check if node_modules exists, if not, run npm install
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("\n====== Installing Frontend Dependencies ======\n")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    try:
        print("\n====== Starting Frontend Development Server ======\n")
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            # Redirect stdout and stderr to current terminal
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        
        # Print output in real-time
        for line in frontend_process.stdout:
            print(f"[FRONTEND] {line.strip()}")
            
            # Open browser when frontend is ready
            if "Local:" in line:
                url = line.split("Local:")[1].strip()
                print(f"\n====== Opening browser at {url} ======\n")
                webbrowser.open(url)
                
        return frontend_process
    except Exception as e:
        print(f"Error starting frontend: {str(e)}")
        return None

def run_backend_thread():
    """Run the backend in a separate thread"""
    run_backend()
    
def run_frontend_thread():
    """Run the frontend in a separate thread"""
    # Wait for backend to start
    time.sleep(5)  
    run_frontend()

if __name__ == "__main__":
    print("Starting Lead Scoring System...")
    
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join("MAIN", "output"), exist_ok=True)
    os.makedirs(os.path.join("MAIN", "models"), exist_ok=True)
    
    # Start backend and frontend in separate threads
    backend_thread = threading.Thread(target=run_backend_thread)
    frontend_thread = threading.Thread(target=run_frontend_thread)
    
    backend_thread.daemon = True
    frontend_thread.daemon = True
    
    backend_thread.start()
    frontend_thread.start()
    
    # Keep main thread running to allow keyboard interrupts
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0) 