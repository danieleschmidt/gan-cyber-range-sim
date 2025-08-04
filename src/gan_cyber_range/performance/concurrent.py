"""Concurrent execution and task management."""

import asyncio
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task representation."""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "duration_ms": self.duration.total_seconds() * 1000 if self.duration else None,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "has_error": self.error is not None
        }


class TaskPool:
    """Task pool for managing concurrent execution."""
    
    def __init__(
        self,
        max_workers: int = None,
        max_queue_size: int = 1000,
        enable_metrics: bool = True
    ):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.max_queue_size = max_queue_size
        self.enable_metrics = enable_metrics
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        # Metrics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.cancelled_tasks = 0
        self.total_execution_time = 0.0
    
    async def start(self) -> None:
        """Start the task pool."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker coroutines
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the task pool."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.wait(self.workers, timeout=timeout)
        
        # Cancel remaining active tasks
        for task_id, task in self.active_tasks.items():
            task.cancel()
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.CANCELLED
        
        self.workers.clear()
        self.active_tasks.clear()
    
    async def submit(
        self,
        func: Callable,
        *args,
        name: str = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        if not self.running:
            await self.start()
        
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.tasks[task_id] = task
        
        # Add to queue (negative priority for max-heap behavior)
        await self.task_queue.put((-priority, task_id))
        
        if self.enable_metrics:
            self.total_tasks += 1
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, waiting if necessary."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for completion
        start_time = time.time()
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out waiting for result")
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise task.error
        elif task.status == TaskStatus.CANCELLED:
            raise asyncio.CancelledError(f"Task {task_id} was cancelled")
        else:
            raise RuntimeError(f"Task {task_id} in unexpected state: {task.status}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        elif task.status == TaskStatus.RUNNING:
            # Cancel the active task
            active_task = self.active_tasks.get(task_id)
            if active_task:
                active_task.cancel()
                task.status = TaskStatus.CANCELLED
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status information."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return task.to_dict()
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """List tasks, optionally filtered by status."""
        tasks = []
        for task in self.tasks.values():
            if status is None or task.status == status:
                tasks.append(task.to_dict())
        
        return sorted(tasks, key=lambda x: x["created_at"])
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine."""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Execute task
                await self._execute_task(worker_name, task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log worker error but continue
                print(f"Worker {worker_name} error: {e}")
    
    async def _execute_task(self, worker_name: str, task: Task) -> None:
        """Execute a single task."""
        async with self.semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                # Create execution task
                if asyncio.iscoroutinefunction(task.func):
                    exec_task = asyncio.create_task(
                        task.func(*task.args, **task.kwargs)
                    )
                else:
                    # Run sync function in thread pool
                    exec_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: task.func(*task.args, **task.kwargs)
                        )
                    )
                
                self.active_tasks[task.id] = exec_task
                
                # Execute with timeout
                if task.timeout:
                    task.result = await asyncio.wait_for(exec_task, timeout=task.timeout)
                else:
                    task.result = await exec_task
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                if self.enable_metrics:
                    self.completed_tasks += 1
                    if task.duration:
                        self.total_execution_time += task.duration.total_seconds()
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                if self.enable_metrics:
                    self.cancelled_tasks += 1
                
            except Exception as e:
                task.error = e
                task.completed_at = datetime.now()
                
                # Retry if configured
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.error = None
                    
                    # Re-queue for retry
                    await self.task_queue.put((-task.priority, task.id))
                else:
                    task.status = TaskStatus.FAILED
                    if self.enable_metrics:
                        self.failed_tasks += 1
            
            finally:
                # Clean up
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get task pool metrics."""
        queue_size = self.task_queue.qsize()
        active_count = len(self.active_tasks)
        
        avg_execution_time = (
            self.total_execution_time / max(self.completed_tasks, 1)
            if self.completed_tasks > 0 else 0
        )
        
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "pending_tasks": queue_size,
            "active_tasks": active_count,
            "max_workers": self.max_workers,
            "avg_execution_time_ms": avg_execution_time * 1000,
            "success_rate": self.completed_tasks / max(self.total_tasks, 1) if self.total_tasks > 0 else 0
        }


class ConcurrentExecutor:
    """High-level concurrent execution manager."""
    
    def __init__(
        self,
        async_pool_size: int = None,
        thread_pool_size: int = None,
        process_pool_size: int = None
    ):
        # Async task pool
        self.async_pool = TaskPool(max_workers=async_pool_size)
        
        # Thread pool for I/O bound tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size or min(32, (multiprocessing.cpu_count() or 1) + 4)
        )
        
        # Process pool for CPU bound tasks
        self.process_pool = ProcessPoolExecutor(
            max_workers=process_pool_size or multiprocessing.cpu_count()
        )
        
        self.running = False
    
    async def start(self) -> None:
        """Start the executor."""
        if not self.running:
            await self.async_pool.start()
            self.running = True
    
    async def stop(self) -> None:
        """Stop the executor."""
        if self.running:
            await self.async_pool.stop()
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            self.running = False
    
    async def submit_async(
        self,
        coro: Coroutine,
        name: str = None,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> str:
        """Submit async coroutine."""
        return await self.async_pool.submit(
            lambda: coro,
            name=name,
            priority=priority,
            timeout=timeout
        )
    
    async def submit_io_bound(
        self,
        func: Callable,
        *args,
        name: str = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Submit I/O bound task to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )
    
    async def submit_cpu_bound(
        self,
        func: Callable,
        *args,
        name: str = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Submit CPU bound task to process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.process_pool,
            lambda: func(*args, **kwargs)
        )
    
    async def map_async(
        self,
        func: Callable,
        iterable: List[Any],
        max_concurrency: int = 10
    ) -> List[Any]:
        """Map function over iterable with controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_func(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    return await self.submit_io_bound(func, item)
        
        tasks = [limited_func(item) for item in iterable]
        return await asyncio.gather(*tasks)
    
    async def gather_with_limit(
        self,
        *coroutines,
        limit: int = 10,
        return_exceptions: bool = False
    ) -> List[Any]:
        """Gather coroutines with concurrency limit."""
        semaphore = asyncio.Semaphore(limit)
        
        async def limited_coro(coro):
            async with semaphore:
                return await coro
        
        limited_coros = [limited_coro(coro) for coro in coroutines]
        return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)
    
    async def batch_process(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 100,
        max_concurrency: int = 5
    ) -> List[Any]:
        """Process items in batches with concurrency control."""
        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches concurrently
        async def process_batch(batch):
            if asyncio.iscoroutinefunction(processor):
                tasks = [processor(item) for item in batch]
                return await asyncio.gather(*tasks)
            else:
                return await self.map_async(processor, batch, max_concurrency)
        
        batch_results = await self.map_async(
            process_batch,
            batches,
            max_concurrency
        )
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics."""
        async_metrics = self.async_pool.get_metrics()
        
        return {
            "async_pool": async_metrics,
            "thread_pool": {
                "max_workers": self.thread_pool._max_workers,
                "active_threads": len(self.thread_pool._threads)
            },
            "process_pool": {
                "max_workers": self.process_pool._max_workers,
                "active_processes": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
            }
        }


class RateLimiter:
    """Rate limiter for controlling execution frequency."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire rate limit token."""
        async with self.lock:
            now = time.time()
            
            # Remove old calls outside time window
            cutoff = now - self.time_window
            self.calls = [call_time for call_time in self.calls if call_time > cutoff]
            
            # Check if we can make a call
            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            # Record this call
            self.calls.append(now)
    
    def __call__(self, func):
        """Decorator to apply rate limiting."""
        async def wrapper(*args, **kwargs):
            await self.acquire()
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper