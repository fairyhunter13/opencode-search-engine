"""
Parent PID monitoring pattern for Python embedder server.

Add this to the embedder's server.py startup logic.
"""

import asyncio
import os
import signal
import logging

async def monitor_parent(parent_pid: int, shutdown_event: asyncio.Event):
    """
    Monitor parent process and trigger shutdown if it dies.
    
    Args:
        parent_pid: Parent process PID to monitor
        shutdown_event: Event to set when parent dies
    """
    logging.info(f"Monitoring parent PID {parent_pid}")
    
    while True:
        await asyncio.sleep(5)
        try:
            # kill(pid, 0) doesn't send a signal, just checks if process exists
            os.kill(parent_pid, 0)
        except OSError:
            # Parent died
            logging.warning(f"Parent process {parent_pid} died, initiating shutdown")
            shutdown_event.set()
            break

def setup_parent_monitor(shutdown_event: asyncio.Event):
    """
    Set up parent monitoring if OPENCODE_EMBEDDER_PARENT_PID is set.
    
    Call this during server initialization, before starting the event loop.
    
    Args:
        shutdown_event: Shutdown event to trigger when parent dies
    
    Returns:
        The monitor task if started, None otherwise
    """
    parent_pid_str = os.environ.get("OPENCODE_EMBEDDER_PARENT_PID")
    if not parent_pid_str:
        return None
    
    try:
        parent_pid = int(parent_pid_str)
    except ValueError:
        logging.warning(f"Invalid OPENCODE_EMBEDDER_PARENT_PID: {parent_pid_str}")
        return None
    
    # Spawn monitor task
    task = asyncio.create_task(monitor_parent(parent_pid, shutdown_event))
    return task


# ============================================================================
# Usage example (add to server.py)
# ============================================================================

# async def main():
#     shutdown_event = asyncio.Event()
#     
#     # Set up parent monitor
#     monitor_task = setup_parent_monitor(shutdown_event)
#     
#     # ... start your FastAPI/Starlette/aiohttp server ...
#     
#     # Wait for shutdown signal
#     await shutdown_event.wait()
#     
#     # Clean shutdown
#     logging.info("Shutdown triggered, cleaning up...")
#     # ... stop server, cleanup resources ...
#     if monitor_task:
#         monitor_task.cancel()
