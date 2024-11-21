import signal

class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        print("\nShutdown requested... Completing current tasks...")
        self.kill_now = True
