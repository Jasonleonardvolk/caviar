"""
File system watcher for ELFIN.

This module provides functionality to watch for file changes on disk
and notify the language server.
"""

import logging
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Set, Optional, Callable, List, Any, Union

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)


class ElfinFileWatcher(FileSystemEventHandler):
    """
    File system watcher for ELFIN files.
    
    This class uses the watchdog library to watch for file changes on disk
    and triggers callbacks when ELFIN files are created, modified, or deleted.
    """
    
    def __init__(
        self,
        on_change: Callable[[str], None],
        on_create: Optional[Callable[[str], None]] = None,
        on_delete: Optional[Callable[[str], None]] = None,
        root_dirs: Optional[List[Union[str, Path]]] = None,
        extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the file watcher.
        
        Args:
            on_change: Callback function for file changes
            on_create: Callback function for file creation (optional)
            on_delete: Callback function for file deletion (optional)
            root_dirs: Directories to watch (optional)
            extensions: File extensions to watch (optional)
        """
        super().__init__()
        
        self.on_change = on_change
        self.on_create = on_create or (lambda path: None)
        self.on_delete = on_delete or (lambda path: None)
        
        self.root_dirs = root_dirs or []
        self.extensions = extensions or [".elfin"]
        
        self.observer = Observer()
        self.is_watching = False
        
        # Debounce variables
        self.debounce_delay = 0.5  # seconds
        self.last_events: Dict[str, float] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task = None
    
    def is_elfin_file(self, path: str) -> bool:
        """
        Check if a file is an ELFIN file.
        
        Args:
            path: Path to the file
            
        Returns:
            True if the file is an ELFIN file, False otherwise
        """
        if not path:
            return False
        
        # Check if the file has a relevant extension
        file_ext = os.path.splitext(path)[1].lower()
        return file_ext in self.extensions
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handle file modification events.
        
        Args:
            event: File system event
        """
        if not event.is_directory and self.is_elfin_file(event.src_path):
            # Schedule the event for processing
            self._schedule_event("change", event.src_path)
    
    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation events.
        
        Args:
            event: File system event
        """
        if not event.is_directory and self.is_elfin_file(event.src_path):
            # Schedule the event for processing
            self._schedule_event("create", event.src_path)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        Handle file deletion events.
        
        Args:
            event: File system event
        """
        if not event.is_directory and self.is_elfin_file(event.src_path):
            # Schedule the event for processing
            self._schedule_event("delete", event.src_path)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """
        Handle file move events.
        
        Args:
            event: File system event
        """
        # When a file is moved, we receive a moved event with both src_path and dest_path
        if not event.is_directory:
            if self.is_elfin_file(event.src_path):
                # Schedule a delete event for the source path
                self._schedule_event("delete", event.src_path)
            
            if hasattr(event, "dest_path") and self.is_elfin_file(event.dest_path):
                # Schedule a create event for the destination path
                self._schedule_event("create", event.dest_path)
    
    def _schedule_event(self, event_type: str, path: str) -> None:
        """
        Schedule an event for processing with debouncing.
        
        Args:
            event_type: Type of event
            path: Path to the file
        """
        # Record the event time
        now = time.time()
        self.last_events[path] = now
        
        # Put the event in the queue for processing
        asyncio.create_task(self.event_queue.put((event_type, path, now)))
        
        # Ensure the processing task is running
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_events())
    
    async def _process_events(self) -> None:
        """Process events with debouncing."""
        while True:
            # Get the next event
            event_type, path, event_time = await self.event_queue.get()
            
            # Sleep to debounce
            await asyncio.sleep(self.debounce_delay)
            
            # Check if this event is still the most recent for this path
            if self.last_events.get(path) == event_time:
                # Process the event
                if event_type == "change":
                    self.on_change(path)
                elif event_type == "create":
                    self.on_create(path)
                elif event_type == "delete":
                    self.on_delete(path)
                
                # Clean up
                del self.last_events[path]
            
            # Mark the event as processed
            self.event_queue.task_done()
            
            # If the queue is empty, exit
            if self.event_queue.empty():
                break
    
    def start_watching(self) -> None:
        """Start watching for file changes."""
        if not self.is_watching:
            for root_dir in self.root_dirs:
                # Convert to Path if it's a string
                if isinstance(root_dir, str):
                    root_dir = Path(root_dir)
                
                # Ensure the directory exists
                if root_dir.exists() and root_dir.is_dir():
                    self.observer.schedule(self, str(root_dir), recursive=True)
                    logger.info(f"Watching directory: {root_dir}")
                else:
                    logger.warning(f"Directory does not exist or is not a directory: {root_dir}")
            
            self.observer.start()
            self.is_watching = True
            logger.info("File watcher started")
    
    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        if self.is_watching:
            self.observer.stop()
            self.observer.join()
            self.is_watching = False
            logger.info("File watcher stopped")
    
    def add_directory(self, directory: Union[str, Path]) -> None:
        """
        Add a directory to watch.
        
        Args:
            directory: Directory to watch
        """
        # Convert to Path if it's a string
        if isinstance(directory, str):
            directory = Path(directory)
        
        # Ensure the directory exists
        if directory.exists() and directory.is_dir():
            # Check if we're already watching this directory
            if directory not in self.root_dirs:
                self.root_dirs.append(directory)
                
                # If we're already watching, schedule the new directory
                if self.is_watching:
                    self.observer.schedule(self, str(directory), recursive=True)
                    logger.info(f"Watching directory: {directory}")
            else:
                logger.debug(f"Already watching directory: {directory}")
        else:
            logger.warning(f"Directory does not exist or is not a directory: {directory}")
