"""Advanced concurrent execution and task management system."""

import asyncio
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from queue import Queue, Empty
import multiprocessing as mp


T = TypeVar('T')


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_seconds": self.execution_time_seconds
        }


@dataclass
class Task:
    """Task definition."""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value  # Higher priority first


class TaskPool:
    """Advanced task pool with priority scheduling and resource management."""
    
    def __init__(
        self,
        max_workers: int = None,
        worker_type: str = "thread",  # "thread" or "process"
        queue_size: int = 1000
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.worker_type = worker_type
        self.queue_size = queue_size
        
        # Task management
        self.task_queue = asyncio.PriorityQueue(maxsize=queue_size)
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Executors
        if worker_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "average_execution_time": 0.0
        }
        
        self.running = False
        self.workers = []
        self.logger = logging.getLogger(f"task_pool_{worker_type}")
    
    async def start(self, num_workers: Optional[int] = None):
        """Start the task pool workers."""
        if self.running:
            return
        
        self.running = True
        worker_count = num_workers or self.max_workers
        
        self.logger.info(f"Starting task pool with {worker_count} {self.worker_type} workers")
        
        # Start worker coroutines
        for i in range(worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self, timeout: float = 30.0):
        """Stop the task pool."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping task pool...")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.wait(self.workers, timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        self.workers.clear()
        
        self.logger.info("Task pool stopped")
    
    async def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        # Add to queue (with priority as negative for proper ordering)
        await self.task_queue.put((-priority.value, time.time(), task))
        self.stats["tasks_submitted"] += 1
        
        self.logger.debug(f"Task submitted: {task_id} (priority: {priority.name})")
        return task_id
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks."""
        self.logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get next task with timeout
                try:
                    _, _, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                await self._execute_task(task, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
        
        self.logger.debug(f"Worker {worker_name} stopped")
    
    async def _execute_task(self, task: Task, worker_name: str):
        """Execute a single task."""
        self.active_tasks[task.task_id] = task
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        try:
            self.logger.debug(f"Worker {worker_name} executing task: {task.task_id}")
            
            # Execute function with timeout
            if asyncio.iscoroutinefunction(task.func):
                # Async function
                if task.timeout_seconds:
                    coro_result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout_seconds
                    )
                else:
                    coro_result = await task.func(*task.args, **task.kwargs)
                result.result = coro_result
            else:
                # Sync function - run in executor
                loop = asyncio.get_event_loop()
                if task.timeout_seconds:
                    future = loop.run_in_executor(
                        self.executor, task.func, *task.args
                    )
                    coro_result = await asyncio.wait_for(future, timeout=task.timeout_seconds)
                else:
                    coro_result = await loop.run_in_executor(
                        self.executor, task.func, *task.args
                    )
                result.result = coro_result
            
            # Task completed successfully
            result.status = TaskStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.execution_time_seconds = (result.end_time - result.start_time).total_seconds()
            
            self.stats["tasks_completed"] += 1
            
            # Update average execution time
            total_tasks = self.stats["tasks_completed"]
            if total_tasks > 1:
                self.stats["average_execution_time"] = (
                    (self.stats["average_execution_time"] * (total_tasks - 1) + 
                     result.execution_time_seconds) / total_tasks
                )
            else:
                self.stats["average_execution_time"] = result.execution_time_seconds
            
            self.logger.debug(
                f"Task {task.task_id} completed in {result.execution_time_seconds:.3f}s"
            )
            
        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error = f"Task timed out after {task.timeout_seconds}s"
            result.end_time = datetime.utcnow()
            self.stats["tasks_failed"] += 1
            
            self.logger.warning(f"Task {task.task_id} timed out")
            
        except asyncio.CancelledError:
            result.status = TaskStatus.CANCELLED
            result.error = "Task was cancelled"
            result.end_time = datetime.utcnow()
            self.stats["tasks_cancelled"] += 1
            
            self.logger.info(f"Task {task.task_id} cancelled")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.utcnow()
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.logger.warning(
                    f"Task {task.task_id} failed (attempt {task.retry_count}/{task.max_retries + 1}): {e}"
                )
                
                # Re-queue with exponential backoff
                await asyncio.sleep(2 ** task.retry_count)
                await self.task_queue.put((-task.priority.value, time.time(), task))
                
                # Remove from active tasks but don't add to completed
                self.active_tasks.pop(task.task_id, None)
                return
            else:
                self.stats["tasks_failed"] += 1
                self.logger.error(
                    f"Task {task.task_id} failed permanently after {task.max_retries + 1} attempts: {e}"
                )
        
        finally:
            # Clean up
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = result
            
            # Keep only last 1000 completed tasks
            if len(self.completed_tasks) > 1000:
                oldest_id = min(self.completed_tasks.keys(), 
                              key=lambda k: self.completed_tasks[k].start_time or datetime.min)
                self.completed_tasks.pop(oldest_id, None)
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a specific task."""
        if task_id in self.active_tasks:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                start_time=datetime.utcnow()
            )
        return self.completed_tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task pool statistics."""
        return {
            "worker_type": self.worker_type,
            "max_workers": self.max_workers,
            "running": self.running,
            "active_workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "stats": self.stats.copy()
        }


class AsyncTaskManager:
    """High-level async task management system."""
    
    def __init__(self):
        self.task_pools: Dict[str, TaskPool] = {}
        self.default_pool_name = "default"
        self.logger = logging.getLogger("async_task_manager")
        
        # Create default pool
        self.create_pool(self.default_pool_name)
    
    def create_pool(
        self,
        pool_name: str,
        max_workers: int = None,
        worker_type: str = "thread"
    ) -> TaskPool:
        """Create a new task pool."""
        if pool_name in self.task_pools:
            raise ValueError(f"Pool '{pool_name}' already exists")
        
        pool = TaskPool(max_workers=max_workers, worker_type=worker_type)
        self.task_pools[pool_name] = pool
        return pool
    
    async def start_all_pools(self):
        """Start all task pools."""
        for pool_name, pool in self.task_pools.items():
            await pool.start()
            self.logger.info(f"Started task pool: {pool_name}")
    
    async def stop_all_pools(self):
        """Stop all task pools."""
        for pool_name, pool in self.task_pools.items():
            await pool.stop()
            self.logger.info(f"Stopped task pool: {pool_name}")
    
    async def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        pool_name: str = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        pool_name = pool_name or self.default_pool_name
        if pool_name not in self.task_pools:
            raise ValueError(f"Pool '{pool_name}' does not exist")
        
        pool = self.task_pools[pool_name]
        return await pool.submit_task(
            task_id=task_id,
            func=func,
            *args,
            priority=priority,
            timeout_seconds=timeout_seconds,
            **kwargs
        )
    
    async def wait_for_task(
        self,
        task_id: str,
        pool_name: str = None,
        timeout: Optional[float] = None
    ) -> TaskResult:
        """Wait for a task to complete."""
        pool_name = pool_name or self.default_pool_name
        pool = self.task_pools[pool_name]
        
        start_time = time.time()
        while True:
            result = pool.get_task_status(task_id)
            if result and result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, 
                                          TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            await asyncio.sleep(0.1)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        return {
            pool_name: pool.get_stats()
            for pool_name, pool in self.task_pools.items()
        }


class BatchProcessor:
    """Batch processing system for handling large numbers of similar tasks."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 5,
        task_manager: Optional[AsyncTaskManager] = None
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.task_manager = task_manager or AsyncTaskManager()
        self.logger = logging.getLogger("batch_processor")
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable[[List[Any]], Awaitable[List[Any]]],
        batch_id: Optional[str] = None
    ) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        self.logger.info(
            f"Processing {len(items)} items in {len(batches)} batches (batch_id: {batch_id})"
        )
        
        # Process batches concurrently with limit
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_single_batch(batch_items: List[Any], batch_index: int):
            async with semaphore:
                task_id = f"{batch_id}_batch_{batch_index}"
                try:
                    return await self.task_manager.submit(
                        processor_func,
                        batch_items,
                        task_id=task_id,
                        priority=TaskPriority.NORMAL
                    )
                except Exception as e:
                    self.logger.error(f"Batch {batch_index} failed: {e}")
                    return []
        
        # Submit all batch tasks
        batch_tasks = [
            process_single_batch(batch, i)
            for i, batch in enumerate(batches)
        ]
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch processing error: {batch_result}")
            elif isinstance(batch_result, list):
                results.extend(batch_result)
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results


# Global task manager instance
global_task_manager = AsyncTaskManager()


# Convenience functions
async def submit_task(
    func: Callable,
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    timeout_seconds: Optional[float] = None,
    **kwargs
) -> str:
    """Submit task to global task manager."""
    return await global_task_manager.submit(
        func, *args, priority=priority, timeout_seconds=timeout_seconds, **kwargs
    )


async def wait_for_result(task_id: str, timeout: Optional[float] = None) -> TaskResult:
    """Wait for task result from global task manager."""
    return await global_task_manager.wait_for_task(task_id, timeout=timeout)


# Example usage
async def example_usage():
    """Example of concurrent execution system."""
    
    # Define some example tasks
    async def async_task(name: str, duration: float) -> str:
        await asyncio.sleep(duration)
        return f"Task {name} completed"
    
    def sync_task(name: str, duration: float) -> str:
        time.sleep(duration)
        return f"Sync task {name} completed"
    
    # Start task manager
    await global_task_manager.start_all_pools()
    
    try:
        # Submit various tasks
        tasks = []
        
        # High priority async task
        task1 = await submit_task(
            async_task, "important", 1.0,
            priority=TaskPriority.HIGH
        )
        tasks.append(task1)
        
        # Normal priority sync task
        task2 = await submit_task(
            sync_task, "normal", 0.5,
            priority=TaskPriority.NORMAL
        )
        tasks.append(task2)
        
        # Low priority task with timeout
        task3 = await submit_task(
            async_task, "slow", 3.0,
            priority=TaskPriority.LOW,
            timeout_seconds=2.0
        )
        tasks.append(task3)
        
        # Wait for all tasks
        results = []
        for task_id in tasks:
            result = await wait_for_result(task_id, timeout=5.0)
            results.append(result)
            print(f"Task {task_id}: {result.status.value} - {result.result or result.error}")
        
        # Show statistics
        stats = global_task_manager.get_all_stats()
        print(f"Task manager stats: {stats}")
        
    finally:
        await global_task_manager.stop_all_pools()


if __name__ == "__main__":
    asyncio.run(example_usage())