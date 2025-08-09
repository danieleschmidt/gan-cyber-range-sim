"""
Distributed Computing Engine for GAN Cyber Range.

Provides scalable distributed processing with:
- Agent workload distribution
- Simulation parallelization  
- GPU cluster management
- Cross-region load balancing
- Fault-tolerant task execution
"""

import asyncio
import logging
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class WorkerType(str, Enum):
    """Worker node types."""
    CPU_WORKER = "cpu_worker"
    GPU_WORKER = "gpu_worker"
    MEMORY_WORKER = "memory_worker"
    NETWORK_WORKER = "network_worker"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class ComputeResource:
    """Compute resource specification."""
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.0
    network_bandwidth_mbps: float = 1000.0


@dataclass
class WorkerNode:
    """Distributed worker node."""
    worker_id: str
    worker_type: WorkerType
    endpoint: str
    region: str
    zone: str
    
    # Resource capacity
    total_resources: ComputeResource
    available_resources: ComputeResource
    
    # Status
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load_factor: float = 0.0
    
    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_task_duration: float = 0.0
    
    # Capabilities
    supported_frameworks: List[str] = field(default_factory=list)
    specialized_functions: List[str] = field(default_factory=list)


@dataclass
class DistributedTask:
    """Distributed computation task."""
    task_id: str
    task_type: str
    priority: TaskPriority
    
    # Task specification
    function_name: str
    parameters: Dict[str, Any]
    required_resources: ComputeResource
    estimated_duration: int  # seconds
    
    # Execution constraints
    max_retries: int = 3
    timeout_seconds: int = 3600
    required_worker_type: Optional[WorkerType] = None
    preferred_regions: List[str] = field(default_factory=list)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass
class TaskBatch:
    """Batch of related tasks."""
    batch_id: str
    name: str
    tasks: List[DistributedTask]
    created_at: datetime = field(default_factory=datetime.now)
    
    # Batch constraints
    max_parallel_tasks: int = 10
    batch_timeout: int = 7200  # seconds
    
    # State tracking
    completed_tasks: int = 0
    failed_tasks: int = 0


class WorkerManager:
    """Manages distributed worker nodes."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_health_check_interval = 30  # seconds
        self.worker_timeout = 120  # seconds
        
        # Load balancing
        self.load_balancing_strategy = "least_loaded"  # least_loaded, round_robin, locality
        self.region_weights: Dict[str, float] = {}
    
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node."""
        self.workers[worker.worker_id] = worker
        logger.info(f"Registered worker {worker.worker_id} in {worker.region}/{worker.zone}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker node."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Unregistered worker {worker_id}")
    
    def get_available_workers(self, 
                             task: DistributedTask,
                             required_resources: Optional[ComputeResource] = None) -> List[WorkerNode]:
        """Get workers that can execute a task."""
        suitable_workers = []
        
        for worker in self.workers.values():
            if not worker.is_healthy:
                continue
            
            # Check worker type constraint
            if task.required_worker_type and worker.worker_type != task.required_worker_type:
                continue
            
            # Check resource availability
            resources = required_resources or task.required_resources
            if not self._has_sufficient_resources(worker, resources):
                continue
            
            # Check regional preferences
            if task.preferred_regions and worker.region not in task.preferred_regions:
                continue
            
            # Check framework support
            if hasattr(task, 'required_framework'):
                if task.required_framework not in worker.supported_frameworks:
                    continue
            
            suitable_workers.append(worker)
        
        return suitable_workers
    
    def select_optimal_worker(self, 
                             task: DistributedTask, 
                             available_workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Select the optimal worker for a task."""
        if not available_workers:
            return None
        
        if self.load_balancing_strategy == "least_loaded":
            return min(available_workers, key=lambda w: w.load_factor)
        
        elif self.load_balancing_strategy == "round_robin":
            # Simple round-robin based on task count
            return min(available_workers, key=lambda w: w.tasks_completed + w.tasks_failed)
        
        elif self.load_balancing_strategy == "locality":
            # Prefer workers in preferred regions
            if task.preferred_regions:
                preferred_workers = [w for w in available_workers if w.region in task.preferred_regions]
                if preferred_workers:
                    return min(preferred_workers, key=lambda w: w.load_factor)
            
            return min(available_workers, key=lambda w: w.load_factor)
        
        return available_workers[0]
    
    def _has_sufficient_resources(self, worker: WorkerNode, required: ComputeResource) -> bool:
        """Check if worker has sufficient resources."""
        available = worker.available_resources
        
        return (available.cpu_cores >= required.cpu_cores and
                available.memory_gb >= required.memory_gb and
                available.gpu_count >= required.gpu_count and
                available.gpu_memory_gb >= required.gpu_memory_gb and
                available.storage_gb >= required.storage_gb)
    
    def update_worker_resources(self, worker_id: str, used_resources: ComputeResource):
        """Update worker resource availability."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        available = worker.available_resources
        
        # Subtract used resources
        available.cpu_cores -= used_resources.cpu_cores
        available.memory_gb -= used_resources.memory_gb
        available.gpu_count -= used_resources.gpu_count
        available.gpu_memory_gb -= used_resources.gpu_memory_gb
        available.storage_gb -= used_resources.storage_gb
        
        # Update load factor
        total = worker.total_resources
        cpu_usage = 1.0 - (available.cpu_cores / total.cpu_cores)
        memory_usage = 1.0 - (available.memory_gb / total.memory_gb)
        worker.load_factor = max(cpu_usage, memory_usage)
    
    def release_worker_resources(self, worker_id: str, used_resources: ComputeResource):
        """Release worker resources after task completion."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        available = worker.available_resources
        total = worker.total_resources
        
        # Add back released resources (but don't exceed total capacity)
        available.cpu_cores = min(available.cpu_cores + used_resources.cpu_cores, total.cpu_cores)
        available.memory_gb = min(available.memory_gb + used_resources.memory_gb, total.memory_gb)
        available.gpu_count = min(available.gpu_count + used_resources.gpu_count, total.gpu_count)
        available.gpu_memory_gb = min(available.gpu_memory_gb + used_resources.gpu_memory_gb, total.gpu_memory_gb)
        available.storage_gb = min(available.storage_gb + used_resources.storage_gb, total.storage_gb)
        
        # Update load factor
        cpu_usage = 1.0 - (available.cpu_cores / total.cpu_cores)
        memory_usage = 1.0 - (available.memory_gb / total.memory_gb)
        worker.load_factor = max(cpu_usage, memory_usage)


class TaskScheduler:
    """Intelligent task scheduler for distributed execution."""
    
    def __init__(self, worker_manager: WorkerManager):
        self.worker_manager = worker_manager
        
        # Task queues by priority
        self.task_queues: Dict[TaskPriority, List[DistributedTask]] = {
            priority: [] for priority in TaskPriority
        }
        
        # Active tasks
        self.active_tasks: Dict[str, DistributedTask] = {}
        
        # Completed tasks
        self.completed_tasks: Dict[str, DistributedTask] = {}
        
        # Task dependencies
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Scheduling configuration
        self.max_concurrent_tasks = 100
        self.scheduling_interval = 5  # seconds
    
    def submit_task(self, task: DistributedTask):
        """Submit a task for execution."""
        # Add to appropriate priority queue
        self.task_queues[task.priority].append(task)
        
        # Update dependency graph
        if task.dependencies:
            self.dependency_graph[task.task_id] = set(task.dependencies)
        
        logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
    
    def submit_batch(self, batch: TaskBatch):
        """Submit a batch of tasks."""
        for task in batch.tasks:
            self.submit_task(task)
        
        logger.info(f"Submitted batch {batch.batch_id} with {len(batch.tasks)} tasks")
    
    def get_ready_tasks(self) -> List[DistributedTask]:
        """Get tasks that are ready to execute (no pending dependencies)."""
        ready_tasks = []
        
        # Process queues by priority (highest first)
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self.task_queues[priority]
            
            for task in queue[:]:  # Copy list to allow modification during iteration
                if self._are_dependencies_satisfied(task):
                    ready_tasks.append(task)
                    queue.remove(task)
        
        return ready_tasks
    
    def _are_dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if all task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def schedule_task(self, task: DistributedTask) -> Optional[WorkerNode]:
        """Schedule a task to an appropriate worker."""
        available_workers = self.worker_manager.get_available_workers(task)
        
        if not available_workers:
            return None
        
        worker = self.worker_manager.select_optimal_worker(task, available_workers)
        
        if worker:
            # Reserve resources
            self.worker_manager.update_worker_resources(worker.worker_id, task.required_resources)
            
            # Update task state
            task.assigned_worker = worker.worker_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Track active task
            self.active_tasks[task.task_id] = task
            
            logger.info(f"Scheduled task {task.task_id} to worker {worker.worker_id}")
        
        return worker
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark a task as completed."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task.completed_at = datetime.now()
        
        if error:
            task.status = TaskStatus.FAILED
            task.error = error
            
            # Check if we should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRY
                task.assigned_worker = None
                task.started_at = None
                
                # Re-queue for retry
                self.task_queues[task.priority].append(task)
                logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
            else:
                logger.error(f"Task {task_id} failed after {task.max_retries} retries: {error}")
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
            
            # Calculate execution time
            if task.started_at:
                task.execution_time = (task.completed_at - task.started_at).total_seconds()
        
        # Release worker resources
        if task.assigned_worker:
            self.worker_manager.release_worker_resources(task.assigned_worker, task.required_resources)
            
            # Update worker performance metrics
            worker = self.worker_manager.workers.get(task.assigned_worker)
            if worker:
                if task.status == TaskStatus.COMPLETED:
                    worker.tasks_completed += 1
                    if task.execution_time:
                        # Update rolling average
                        total_time = worker.avg_task_duration * (worker.tasks_completed - 1) + task.execution_time
                        worker.avg_task_duration = total_time / worker.tasks_completed
                else:
                    worker.tasks_failed += 1
        
        # Move to completed tasks
        del self.active_tasks[task_id]
        self.completed_tasks[task_id] = task
        
        # Update dependent tasks
        self._update_dependents(task_id)
    
    def _update_dependents(self, completed_task_id: str):
        """Update tasks that depend on the completed task."""
        # Remove the completed task from dependency graphs
        for task_id, dependencies in self.dependency_graph.items():
            dependencies.discard(completed_task_id)
    
    def cancel_task(self, task_id: str):
        """Cancel a task."""
        # Remove from queues
        for queue in self.task_queues.values():
            queue[:] = [task for task in queue if task.task_id != task_id]
        
        # Cancel if active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Release resources
            if task.assigned_worker:
                self.worker_manager.release_worker_resources(task.assigned_worker, task.required_resources)
            
            # Move to completed
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            logger.info(f"Cancelled task {task_id}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "pending_tasks": {
                priority.name: len(queue) 
                for priority, queue in self.task_queues.items()
            },
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_workers": len(self.worker_manager.workers),
            "healthy_workers": sum(1 for w in self.worker_manager.workers.values() if w.is_healthy)
        }


class DistributedComputingEngine:
    """Main distributed computing engine."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.worker_manager = WorkerManager()
        self.task_scheduler = TaskScheduler(self.worker_manager)
        self.metrics_collector = metrics_collector
        
        # Runtime state
        self.running = False
        self.engine_tasks: List[asyncio.Task] = []
        
        # Task execution handlers
        self.task_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.cleanup_interval = 300  # seconds
        
        # Performance tracking
        self.throughput_history: List[float] = []
        self.latency_history: List[float] = []
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def start_engine(self):
        """Start the distributed computing engine."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting distributed computing engine")
        
        # Start scheduling loop
        schedule_task = asyncio.create_task(self._scheduling_loop())
        self.engine_tasks.append(schedule_task)
        
        # Start worker health monitoring
        health_task = asyncio.create_task(self._worker_health_monitor())
        self.engine_tasks.append(health_task)
        
        # Start performance monitoring
        perf_task = asyncio.create_task(self._performance_monitor())
        self.engine_tasks.append(perf_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_completed_tasks())
        self.engine_tasks.append(cleanup_task)
        
        logger.info("Distributed computing engine started")
    
    async def stop_engine(self):
        """Stop the distributed computing engine."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping distributed computing engine")
        
        # Cancel all engine tasks
        for task in self.engine_tasks:
            task.cancel()
        
        await asyncio.gather(*self.engine_tasks, return_exceptions=True)
        self.engine_tasks.clear()
        
        logger.info("Distributed computing engine stopped")
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        while self.running:
            try:
                # Get ready tasks
                ready_tasks = self.task_scheduler.get_ready_tasks()
                
                # Schedule as many tasks as possible
                scheduled_count = 0
                for task in ready_tasks:
                    if len(self.task_scheduler.active_tasks) >= self.task_scheduler.max_concurrent_tasks:
                        break
                    
                    worker = self.task_scheduler.schedule_task(task)
                    if worker:
                        # Execute task asynchronously
                        asyncio.create_task(self._execute_task(task, worker))
                        scheduled_count += 1
                
                if scheduled_count > 0:
                    logger.debug(f"Scheduled {scheduled_count} tasks")
                
                await asyncio.sleep(self.task_scheduler.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(self.task_scheduler.scheduling_interval)
    
    async def _execute_task(self, task: DistributedTask, worker: WorkerNode):
        """Execute a task on a worker."""
        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type}")
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                handler(task.function_name, task.parameters, worker),
                timeout=task.timeout_seconds
            )
            
            # Mark as completed
            self.task_scheduler.complete_task(task.task_id, result=result)
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            error_msg = f"Task timeout after {task.timeout_seconds} seconds"
            self.task_scheduler.complete_task(task.task_id, error=error_msg)
            logger.error(f"Task {task.task_id} timed out")
            
        except Exception as e:
            error_msg = str(e)
            self.task_scheduler.complete_task(task.task_id, error=error_msg)
            logger.error(f"Task {task.task_id} failed: {error_msg}")
    
    async def _worker_health_monitor(self):
        """Monitor worker health."""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.worker_manager.worker_timeout)
                
                for worker in self.worker_manager.workers.values():
                    # Check if worker has timed out
                    if worker.last_heartbeat < timeout_threshold:
                        if worker.is_healthy:
                            worker.is_healthy = False
                            logger.warning(f"Worker {worker.worker_id} marked as unhealthy (no heartbeat)")
                    
                    # Update metrics
                    if self.metrics_collector:
                        await self.metrics_collector.record_worker_status(
                            worker_id=worker.worker_id,
                            is_healthy=worker.is_healthy,
                            load_factor=worker.load_factor,
                            tasks_completed=worker.tasks_completed
                        )
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in worker health monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _performance_monitor(self):
        """Monitor system performance."""
        while self.running:
            try:
                # Calculate throughput (tasks completed per minute)
                completed_tasks = len(self.task_scheduler.completed_tasks)
                self.throughput_history.append(completed_tasks)
                
                # Keep only last hour of data
                if len(self.throughput_history) > 60:
                    self.throughput_history = self.throughput_history[-60:]
                
                # Calculate average latency
                recent_tasks = list(self.task_scheduler.completed_tasks.values())[-100:]
                if recent_tasks:
                    avg_latency = sum(task.execution_time for task in recent_tasks if task.execution_time) / len(recent_tasks)
                    self.latency_history.append(avg_latency)
                    
                    # Keep only last hour of data
                    if len(self.latency_history) > 60:
                        self.latency_history = self.latency_history[-60:]
                
                # Update metrics
                if self.metrics_collector:
                    await self.metrics_collector.record_distributed_performance(
                        throughput=self.throughput_history[-1] if self.throughput_history else 0,
                        avg_latency=self.latency_history[-1] if self.latency_history else 0,
                        active_workers=sum(1 for w in self.worker_manager.workers.values() if w.is_healthy)
                    )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_completed_tasks(self):
        """Clean up old completed tasks."""
        while self.running:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(hours=24)  # Keep 24 hours of history
                
                tasks_to_remove = []
                for task_id, task in self.task_scheduler.completed_tasks.items():
                    if task.completed_at and task.completed_at < cleanup_threshold:
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.task_scheduler.completed_tasks[task_id]
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def submit_simulation_task(self, 
                              simulation_id: str,
                              agent_type: str,
                              action_data: Dict[str, Any],
                              priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a simulation task for distributed execution."""
        task_id = f"sim_{simulation_id}_{agent_type}_{uuid.uuid4().hex[:8]}"
        
        # Determine resource requirements based on task complexity
        if "llm_generation" in action_data.get("action_type", ""):
            required_resources = ComputeResource(cpu_cores=2, memory_gb=4.0, gpu_count=1, gpu_memory_gb=8.0)
        else:
            required_resources = ComputeResource(cpu_cores=1, memory_gb=2.0)
        
        task = DistributedTask(
            task_id=task_id,
            task_type="simulation_action",
            priority=priority,
            function_name=f"execute_{agent_type}_action",
            parameters={
                "simulation_id": simulation_id,
                "agent_type": agent_type,
                "action_data": action_data
            },
            required_resources=required_resources,
            estimated_duration=300,  # 5 minutes
            max_retries=2,
            timeout_seconds=600
        )
        
        self.task_scheduler.submit_task(task)
        return task_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        queue_status = self.task_scheduler.get_queue_status()
        
        worker_stats = {
            "total_workers": len(self.worker_manager.workers),
            "healthy_workers": sum(1 for w in self.worker_manager.workers.values() if w.is_healthy),
            "workers_by_type": {},
            "workers_by_region": {},
            "total_capacity": {
                "cpu_cores": 0,
                "memory_gb": 0.0,
                "gpu_count": 0
            },
            "available_capacity": {
                "cpu_cores": 0,
                "memory_gb": 0.0,
                "gpu_count": 0
            }
        }
        
        for worker in self.worker_manager.workers.values():
            # Count by type
            worker_type = worker.worker_type.value
            worker_stats["workers_by_type"][worker_type] = worker_stats["workers_by_type"].get(worker_type, 0) + 1
            
            # Count by region
            region = worker.region
            worker_stats["workers_by_region"][region] = worker_stats["workers_by_region"].get(region, 0) + 1
            
            # Aggregate capacity
            worker_stats["total_capacity"]["cpu_cores"] += worker.total_resources.cpu_cores
            worker_stats["total_capacity"]["memory_gb"] += worker.total_resources.memory_gb
            worker_stats["total_capacity"]["gpu_count"] += worker.total_resources.gpu_count
            
            worker_stats["available_capacity"]["cpu_cores"] += worker.available_resources.cpu_cores
            worker_stats["available_capacity"]["memory_gb"] += worker.available_resources.memory_gb
            worker_stats["available_capacity"]["gpu_count"] += worker.available_resources.gpu_count
        
        performance_stats = {
            "current_throughput": self.throughput_history[-1] if self.throughput_history else 0,
            "avg_throughput": sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0,
            "current_latency": self.latency_history[-1] if self.latency_history else 0,
            "avg_latency": sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        }
        
        return {
            "running": self.running,
            "queue_status": queue_status,
            "worker_stats": worker_stats,
            "performance_stats": performance_stats,
            "registered_task_types": list(self.task_handlers.keys())
        }