# rag_anywhere/server/state.py
import json
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum


class ServerStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    SLEEPING = "sleeping"
    CRASHED = "crashed"


class ServerState:
    """Manages server state persistence and validation"""
    
    def __init__(self, config_dir: Path):
        self.state_file = config_dir / "server.json"
        self.pid_file = config_dir / "server.pid"
    
    def save_state(
        self,
        pid: int,
        port: int,
        active_db: Optional[str],
        status: ServerStatus,
        embedding_model: Optional[str] = None
    ):
        """Save current server state"""
        state = {
            'pid': pid,
            'port': port,
            'active_db': active_db,
            'status': status.value,
            'embedding_model': embedding_model,
            'last_activity': datetime.now().isoformat(),
            'started_at': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load server state if exists"""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def update_activity(self):
        """Update last activity timestamp"""
        state = self.load_state()
        if state:
            state['last_activity'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
    
    def update_status(self, status: ServerStatus):
        """Update server status"""
        state = self.load_state()
        if state:
            state['status'] = status.value
            state['last_activity'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
    
    def clear_state(self):
        """Clear server state files"""
        if self.state_file.exists():
            self.state_file.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def is_server_running(self) -> bool:
        """Check if server process is actually running"""
        state = self.load_state()
        if not state:
            return False
        
        pid = state.get('pid')
        if not pid:
            return False
        
        try:
            process = psutil.Process(pid)
            # Check if it's actually our server process
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def get_actual_status(self) -> ServerStatus:
        """Get actual server status (checking if process is alive)"""
        state = self.load_state()
        
        if not state:
            return ServerStatus.STOPPED
        
        if not self.is_server_running():
            # Process dead but state file exists = crashed
            if state.get('status') in ['running', 'sleeping']:
                return ServerStatus.CRASHED
            return ServerStatus.STOPPED
        
        # Check if should be sleeping
        if state.get('status') == ServerStatus.SLEEPING.value:
            return ServerStatus.SLEEPING
        
        return ServerStatus.RUNNING
    
    def should_wake_from_sleep(self, timeout_minutes: int = 10) -> bool:
        """Check if enough time has passed to sleep"""
        state = self.load_state()
        if not state or state.get('status') != ServerStatus.RUNNING.value:
            return False
        
        last_activity = datetime.fromisoformat(state['last_activity'])
        elapsed = datetime.now() - last_activity
        
        return elapsed > timedelta(minutes=timeout_minutes)
