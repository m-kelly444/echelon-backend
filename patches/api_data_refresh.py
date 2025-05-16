import os
import signal
import time
import threading

class DataRefreshHandler:
    """
    Handler for refreshing data in the API without restart.
    Monitors a trigger file and reloads data when it changes.
    """
    def __init__(self, reload_function, check_interval=60, trigger_file='.reload_data'):
        self.reload_function = reload_function
        self.check_interval = check_interval
        self.trigger_file = trigger_file
        self.last_mtime = 0
        self.running = False
        self.thread = None
    
    def setup_signal_handler(self):
        """Set up a signal handler for SIGUSR1 to reload data"""
        signal.signal(signal.SIGUSR1, self._handle_signal)
        print(f"Signal handler registered for data refresh (PID: {os.getpid()})")
    
    def _handle_signal(self, signum, frame):
        """Handle SIGUSR1 signal by reloading data"""
        if signum == signal.SIGUSR1:
            print("Received reload signal, refreshing data...")
            self.reload_function()
    
    def _check_trigger_file(self):
        """Check if the trigger file has been modified"""
        if os.path.exists(self.trigger_file):
            try:
                mtime = os.path.getmtime(self.trigger_file)
                if mtime > self.last_mtime:
                    print(f"Trigger file modified, refreshing data...")
                    self.reload_function()
                    self.last_mtime = mtime
            except Exception as e:
                print(f"Error checking trigger file: {e}")
    
    def _monitor_loop(self):
        """Background thread to periodically check for data refresh triggers"""
        self.last_mtime = os.path.getmtime(self.trigger_file) if os.path.exists(self.trigger_file) else 0
        
        while self.running:
            self._check_trigger_file()
            time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.thread is not None and self.thread.is_alive():
            print("Monitoring thread already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"Started monitoring for data refreshes (checking every {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("Stopped monitoring for data refreshes")
