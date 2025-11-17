# rag_anywhere/server/manager.py
import subprocess
import time
import sys
from pathlib import Path
from typing import Optional
import socket

from .state import ServerState, ServerStatus
from ..config import Config
from ..utils import get_logger

logger = get_logger('server.manager')


class ServerManager:
    """Manages the RAG Anywhere server lifecycle"""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = ServerState(config.config_dir)
        self.log_dir = config.config_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def is_port_available(self, port: int) -> bool:
        """Check if port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return True
        except OSError:
            return False
    
    def get_configured_port(self) -> int:
        """Get configured port or default"""
        global_config = self.config.load_global_config()
        return global_config.get('server', {}).get('port', 8000)
    
    def start_server(self, port: Optional[int] = None, force: bool = False, debug: bool = False) -> bool:
        """
        Start the server
        
        Args:
            port: Port to use (uses config default if None)
            force: Force restart if already running
            debug: Run in debug mode with verbose output
            
        Returns:
            True if started successfully
        """
        # Check if already running
        if self.state.is_server_running() and not force:
            logger.info("Server already running")
            return True
        
        # Determine port
        if port is None:
            port = self.get_configured_port()
        
        logger.debug(f"Attempting to start server on port {port}")
        
        # Check port availability
        if not self.is_port_available(port):
            raise RuntimeError(
                f"Port {port} is already in use. "
                f"Please stop the service using this port or configure a different port with:\n"
                f"  rag-anywhere config set server.port <new-port>"
            )
        
        # Get active database
        active_db = self.config.get_active_database()
        if not active_db:
            raise ValueError("No active database. Create or activate a database first.")
        
        logger.info(f"Starting server for database '{active_db}'")
        
        # Get embedding model for active database
        db_config = self.config.load_database_config(active_db)
        embedding_model = f"{db_config['embedding']['provider']}:{db_config['embedding']['model']}"
        
        # Setup log files
        stdout_log = self.log_dir / "server-stdout.log"
        stderr_log = self.log_dir / "server-stderr.log"
        
        logger.debug(f"Server logs: stdout={stdout_log}, stderr={stderr_log}")
        
        # Open log files
        stdout_file = open(stdout_log, 'a')
        stderr_file = open(stderr_log, 'a')
        
        # Write separator
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        stdout_file.write(f"\n{'='*60}\n")
        stdout_file.write(f"Server start attempt: {timestamp}\n")
        stdout_file.write(f"Database: {active_db}\n")
        stdout_file.write(f"Port: {port}\n")
        stdout_file.write(f"{'='*60}\n\n")
        stdout_file.flush()
        
        # Start server process
        try:
            # Run as module instead of script to support relative imports
            process = subprocess.Popen(
                [
                    sys.executable, 
                    '-m', 
                    'rag_anywhere.server.app',
                    '--port', str(port), 
                    '--db', active_db
                ],
                stdout=stdout_file if not debug else None,
                stderr=stderr_file if not debug else None,
                start_new_session=True  # Detach from parent
            )
            
            logger.debug(f"Server process started with PID {process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            stdout_file.close()
            stderr_file.close()
            raise RuntimeError(f"Failed to start server process: {e}")
        
        # Wait a moment and check if it started
        logger.debug("Waiting for server to start...")
        time.sleep(3)
        
        # Check if process is still alive
        if process.poll() is not None:
            # Process died
            exit_code = process.returncode
            logger.error(f"Server process died with exit code {exit_code}")
            
            # Read error logs
            stdout_file.close()
            stderr_file.close()
            
            error_msg = "Server failed to start. "
            
            # Try to read last few lines of stderr
            try:
                with open(stderr_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        error_msg += "Last error:\n" + ''.join(lines[-10:])
            except Exception:
                pass
            
            raise RuntimeError(error_msg)
        
        # Verify server is responding (with retries)
        logger.debug("Checking if server is responding...")
        import requests
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=3)
                if response.status_code == 200:
                    logger.debug("Server is responding")
                    break
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Server not ready yet (attempt {attempt + 1}/{max_retries}), waiting...")
                    time.sleep(retry_delay)
                else:
                    # Only log as warning in debug mode, not an error
                    if debug:
                        logger.warning(f"Server may not be fully ready yet: {e}")
                    # Don't fail - server is running, just may be loading
        
        # Save state
        self.state.save_state(
            pid=process.pid,
            port=port,
            active_db=active_db,
            status=ServerStatus.RUNNING,
            embedding_model=embedding_model
        )
        
        logger.info(f"Server started successfully on port {port} (PID: {process.pid})")
        
        # Close log files (process keeps them open)
        stdout_file.close()
        stderr_file.close()
        
        return True
    
    def stop_server(self) -> bool:
        """Stop the server"""
        state_data = self.state.load_state()
        
        if not state_data:
            logger.debug("No server state found")
            return False
        
        pid = state_data.get('pid')
        if not pid:
            logger.debug("No PID in state")
            return False
        
        logger.info(f"Stopping server (PID: {pid})")
        
        try:
            import signal
            import os
            os.kill(pid, signal.SIGTERM)
            
            # Wait for graceful shutdown
            time.sleep(2)
            
            # Force kill if still alive
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already dead
            
            self.state.clear_state()
            logger.info("Server stopped")
            return True
        except ProcessLookupError:
            # Process already dead
            logger.debug("Process already dead")
            self.state.clear_state()
            return False
    
    def restart_server(self, port: Optional[int] = None, debug: bool = False) -> bool:
        """Restart the server"""
        logger.info("Restarting server")
        self.stop_server()
        time.sleep(1)
        return self.start_server(port=port, force=True, debug=debug)
    
    def get_status(self) -> dict:
        """Get current server status"""
        state_data = self.state.load_state()
        actual_status = self.state.get_actual_status()
        
        return {
            'status': actual_status.value,
            'pid': state_data.get('pid') if state_data else None,
            'port': state_data.get('port') if state_data else None,
            'active_db': state_data.get('active_db') if state_data else None,
            'embedding_model': state_data.get('embedding_model') if state_data else None,
            'last_activity': state_data.get('last_activity') if state_data else None,
        }
    
    def ensure_server_running(self, debug: bool = False) -> bool:
        """Ensure server is running, start if needed"""
        status = self.state.get_actual_status()
        
        logger.debug(f"Current server status: {status}")
        
        if status == ServerStatus.STOPPED:
            logger.info("Server not running, starting...")
            return self.start_server(debug=debug)
        elif status == ServerStatus.CRASHED:
            raise RuntimeError(
                "Server has crashed. Please check logs and restart with:\n"
                "  rag-anywhere server restart"
            )
        elif status == ServerStatus.SLEEPING:
            # Wake from sleep
            logger.info("Waking server from sleep")
            self.state.update_status(ServerStatus.RUNNING)
            return True
        
        return True
    
    def switch_database(self, new_db_name: str) -> bool:
        """
        Switch to a different database
        
        This will:
        1. Check if embedding model changed
        2. Reload resources if needed
        3. Always reload FAISS for data isolation
        """
        state_data = self.state.load_state()
        
        if not state_data or not self.state.is_server_running():
            # Server not running, will be started fresh with new DB
            return False
        
        # Get embedding models
        old_db = state_data.get('active_db')
        old_model = state_data.get('embedding_model')
        
        new_db_config = self.config.load_database_config(new_db_name)
        new_model = f"{new_db_config['embedding']['provider']}:{new_db_config['embedding']['model']}"
        
        # Determine if we need to reload model
        reload_model = (old_model != new_model)
        
        logger.info(f"Switching database from '{old_db}' to '{new_db_name}' (reload_model={reload_model})")
        
        # Send reload signal to server
        import requests
        try:
            port = state_data['port']
            response = requests.post(
                f"http://127.0.0.1:{port}/admin/reload",
                json={
                    'database': new_db_name,
                    'reload_model': reload_model
                },
                timeout=30
            )
            
            if response.status_code == 200:
                # Update state
                self.state.save_state(
                    pid=state_data['pid'],
                    port=port,
                    active_db=new_db_name,
                    status=ServerStatus.RUNNING,
                    embedding_model=new_model
                )
                logger.info("Database switched successfully")
                return True
            else:
                logger.error(f"Server returned error: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to communicate with server: {e}")
            return False
