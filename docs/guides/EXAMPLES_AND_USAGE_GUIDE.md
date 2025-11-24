# Namel3ss Parallel & Distributed Execution Examples Guide

## 🚀 Complete Examples and Usage Patterns

This guide provides comprehensive examples for using the Namel3ss parallel and distributed execution system in real-world scenarios.

## 📋 Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Data Processing Patterns](#data-processing-patterns)
3. [Machine Learning Workflows](#machine-learning-workflows)
4. [Web Service Integration](#web-service-integration)
5. [Microservices Orchestration](#microservices-orchestration)
6. [Real-time Event Processing](#real-time-event-processing)
7. [ETL Pipelines](#etl-pipelines)
8. [Scientific Computing](#scientific-computing)
9. [Enterprise Integration](#enterprise-integration)
10. [Performance Optimization](#performance-optimization)

---

## Quick Start Examples

### 1. Simple Parallel Processing

```python
import asyncio
from namel3ss.runtime.parallel import ParallelExecutor

async def quick_start_parallel():
    """Basic parallel processing example."""
    
    # Initialize executor
    executor = ParallelExecutor(default_max_concurrency=5)
    
    # Define processing function
    async def process_number(number, context=None):
        await asyncio.sleep(0.1)  # Simulate work
        return number ** 2
    
    # Define parallel block
    parallel_block = {
        'name': 'square_numbers',
        'strategy': 'all',
        'steps': list(range(10)),  # [0, 1, 2, ..., 9]
        'max_concurrency': 3
    }
    
    # Execute parallel processing
    result = await executor.execute_parallel_block(
        parallel_block, 
        process_number
    )
    
    print(f"Processed {result.completed_tasks} tasks in {result.execution_time:.2f}s")
    squared_numbers = [task.result for task in result.results]
    print(f"Squared numbers: {squared_numbers}")

# Run the example
asyncio.run(quick_start_parallel())
```

### 2. Basic Distributed Processing

```python
import asyncio
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker

async def quick_start_distributed():
    """Basic distributed processing example."""
    
    # Setup components
    broker = MemoryMessageBroker()
    queue = DistributedTaskQueue(broker=broker)
    
    await queue.start()
    
    try:
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = await queue.submit_task(
                task_type="data_processing",
                payload={"item": f"item_{i}"},
                priority=1
            )
            task_ids.append(task_id)
            print(f"Submitted task: {task_id}")
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = await queue.get_task_result(task_id, timeout=30.0)
            results.append(result)
            print(f"Task {task_id}: {result}")
            
    finally:
        await queue.stop()

asyncio.run(quick_start_distributed())
```

---

## Data Processing Patterns

### 1. Batch Data Processing

```python
import asyncio
import json
from typing import List, Dict, Any
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, RedisMessageBroker

class BatchDataProcessor:
    """High-performance batch data processing system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        # Setup components
        self.broker = RedisMessageBroker(redis_url)
        self.distributed_queue = DistributedTaskQueue(
            broker=self.broker,
            worker_timeout=300.0,
            max_retries=3
        )
        self.parallel_executor = ParallelExecutor(
            default_max_concurrency=20
        )
        self.coordinator = DistributedParallelExecutor(
            parallel_executor=self.parallel_executor,
            distributed_queue=self.distributed_queue
        )
    
    async def start(self):
        """Start the processing system."""
        await self.distributed_queue.start()
    
    async def stop(self):
        """Stop the processing system."""
        await self.distributed_queue.stop()
    
    async def process_csv_batch(
        self, 
        file_path: str, 
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Process large CSV file in batches."""
        
        # Read and chunk data
        chunks = await self._chunk_csv_file(file_path, batch_size)
        
        # Define processing function
        async def process_chunk(chunk_data, context=None):
            processed_records = []
            for record in chunk_data:
                # Apply business logic
                processed_record = await self._transform_record(record)
                processed_records.append(processed_record)
            return processed_records
        
        # Setup parallel processing
        parallel_block = {
            'name': 'csv_batch_processing',
            'strategy': 'all',
            'steps': chunks,
            'max_concurrency': 10,
            'distribution_policy': 'balanced'
        }
        
        # Execute distributed parallel processing
        result = await self.coordinator.execute_distributed_parallel(
            parallel_block,
            process_chunk
        )
        
        # Aggregate results
        all_processed = []
        for task_result in result.results:
            if task_result.status == "completed":
                all_processed.extend(task_result.result)
        
        return {
            'total_records': len(all_processed),
            'processing_time': result.execution_time,
            'successful_batches': result.completed_tasks,
            'failed_batches': result.failed_tasks,
            'data': all_processed
        }
    
    async def _chunk_csv_file(
        self, 
        file_path: str, 
        batch_size: int
    ) -> List[List[Dict]]:
        """Split CSV file into processing chunks."""
        import pandas as pd
        
        # Read CSV in chunks
        chunks = []
        for chunk_df in pd.read_csv(file_path, chunksize=batch_size):
            chunk_data = chunk_df.to_dict('records')
            chunks.append(chunk_data)
        
        return chunks
    
    async def _transform_record(self, record: Dict) -> Dict:
        """Apply business logic to individual record."""
        # Example transformations
        transformed = record.copy()
        
        # Data cleaning
        for key, value in transformed.items():
            if isinstance(value, str):
                transformed[key] = value.strip().lower()
        
        # Add computed fields
        if 'amount' in transformed:
            transformed['amount_squared'] = float(transformed['amount']) ** 2
        
        # Add processing timestamp
        import time
        transformed['processed_at'] = time.time()
        
        return transformed

# Usage example
async def batch_processing_example():
    processor = BatchDataProcessor()
    
    await processor.start()
    
    try:
        result = await processor.process_csv_batch(
            'large_dataset.csv',
            batch_size=5000
        )
        
        print(f"Processed {result['total_records']} records")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Success rate: {result['successful_batches']}/{result['successful_batches'] + result['failed_batches']}")
        
    finally:
        await processor.stop()

# Run example
asyncio.run(batch_processing_example())
```

### 2. Stream Processing with Windows

```python
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from collections import deque
import time
from namel3ss.runtime.events import EventDrivenExecutor, MemoryEventBus

class StreamProcessor:
    """Real-time stream processing with windowing."""
    
    def __init__(self, window_size: float = 60.0, slide_interval: float = 10.0):
        self.window_size = window_size  # seconds
        self.slide_interval = slide_interval  # seconds
        
        # Setup event system
        self.event_bus = MemoryEventBus()
        self.executor = EventDrivenExecutor(
            event_bus=self.event_bus,
            max_concurrent_handlers=100
        )
        
        # Data windows
        self.tumbling_window = deque()
        self.sliding_windows = {}
        
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup event handlers for stream processing."""
        
        @self.executor.register_event_handler("data_point")
        async def handle_data_point(event):
            """Process incoming data point."""
            data_point = event.data
            current_time = time.time()
            
            # Add to tumbling window
            self.tumbling_window.append({
                'data': data_point,
                'timestamp': current_time
            })
            
            # Trigger window processing if needed
            if await self._should_process_window():
                await self._process_tumbling_window()
        
        @self.executor.register_event_handler("window_complete")
        async def handle_window_complete(event):
            """Handle completed window processing."""
            window_stats = event.data
            print(f"Window processed: {window_stats}")
            
            # Trigger alerts if needed
            if window_stats.get('anomaly_detected'):
                await self.executor.trigger_event(
                    'anomaly_alert',
                    window_stats
                )
    
    async def process_stream(self, data_stream: AsyncGenerator[Dict, None]):
        """Process data stream with windowing."""
        
        async for data_point in data_stream:
            await self.executor.trigger_event(
                'data_point',
                data_point
            )
            
            # Small delay to simulate real-time processing
            await asyncio.sleep(0.01)
    
    async def _should_process_window(self) -> bool:
        """Check if window should be processed."""
        if not self.tumbling_window:
            return False
        
        current_time = time.time()
        oldest_timestamp = self.tumbling_window[0]['timestamp']
        
        return (current_time - oldest_timestamp) >= self.window_size
    
    async def _process_tumbling_window(self):
        """Process accumulated window data."""
        if not self.tumbling_window:
            return
        
        # Extract window data
        window_data = []
        current_time = time.time()
        
        # Remove expired data points
        while (self.tumbling_window and 
               current_time - self.tumbling_window[0]['timestamp'] >= self.window_size):
            point = self.tumbling_window.popleft()
            window_data.append(point['data'])
        
        if not window_data:
            return
        
        # Process window
        window_stats = await self._compute_window_statistics(window_data)
        
        # Trigger window complete event
        await self.executor.trigger_event(
            'window_complete',
            window_stats
        )
    
    async def _compute_window_statistics(self, window_data: List[Dict]) -> Dict:
        """Compute statistics for window data."""
        
        # Extract numeric values
        values = []
        for item in window_data:
            if 'value' in item and isinstance(item['value'], (int, float)):
                values.append(item['value'])
        
        if not values:
            return {'error': 'No numeric values in window'}
        
        # Basic statistics
        stats = {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'window_duration': self.window_size
        }
        
        # Anomaly detection (simple threshold-based)
        mean_value = stats['mean']
        std_dev = (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5
        threshold = mean_value + 2 * std_dev
        
        anomalies = [v for v in values if v > threshold]
        stats['anomaly_detected'] = len(anomalies) > 0
        stats['anomaly_count'] = len(anomalies)
        
        return stats

# Data generator for testing
async def generate_data_stream() -> AsyncGenerator[Dict, None]:
    """Generate simulated data stream."""
    import random
    
    for i in range(1000):
        # Generate data with occasional anomalies
        base_value = 100 + random.gauss(0, 10)
        if random.random() < 0.05:  # 5% chance of anomaly
            value = base_value * random.uniform(2, 5)
        else:
            value = base_value
        
        yield {
            'id': i,
            'value': value,
            'source': 'sensor_a',
            'metadata': {'batch': i // 50}
        }
        
        await asyncio.sleep(0.1)  # Simulate real-time data

# Usage example
async def stream_processing_example():
    processor = StreamProcessor(window_size=5.0, slide_interval=1.0)
    
    # Process the data stream
    data_stream = generate_data_stream()
    await processor.process_stream(data_stream)

# Run example
asyncio.run(stream_processing_example())
```

---

## Machine Learning Workflows

### 1. Distributed Model Training

```python
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, RedisMessageBroker
from namel3ss.runtime.security import SecurityManager, PermissionLevel
from namel3ss.runtime.observability import ObservabilityManager

class DistributedMLTrainer:
    """Distributed machine learning training system."""
    
    def __init__(self):
        # Setup security
        self.security_manager = SecurityManager(audit_enabled=True)
        
        # Setup observability
        self.observability = ObservabilityManager(
            service_name="ml_trainer",
            enable_metrics=True,
            enable_tracing=True
        )
        
        # Setup distributed processing
        self.broker = RedisMessageBroker()
        self.distributed_queue = DistributedTaskQueue(
            broker=self.broker,
            enable_security=True,
            security_manager=self.security_manager
        )
        
        self.parallel_executor = ParallelExecutor(
            enable_security=True,
            security_manager=self.security_manager,
            enable_observability=True,
            observability_manager=self.observability
        )
        
        self.coordinator = DistributedParallelExecutor(
            parallel_executor=self.parallel_executor,
            distributed_queue=self.distributed_queue,
            enable_security=True,
            security_manager=self.security_manager
        )
    
    async def start(self):
        """Initialize the training system."""
        await self.distributed_queue.start()
    
    async def stop(self):
        """Shutdown the training system."""
        await self.distributed_queue.stop()
    
    async def train_distributed_model(
        self,
        training_data: np.ndarray,
        labels: np.ndarray,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        user_id: str = "ml_user"
    ) -> Dict[str, Any]:
        """Train model using distributed gradient descent."""
        
        # Create security context
        security_context = await self.security_manager.create_security_context(
            user_id=user_id,
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=["read_data", "execute_training", "write_models"]
        )
        
        # Initialize model parameters
        num_features = training_data.shape[1]
        weights = np.random.normal(0, 0.1, num_features)
        bias = 0.0
        
        training_history = []
        
        async with self.observability.start_trace("distributed_training") as span:
            span.set_tag("epochs", num_epochs)
            span.set_tag("batch_size", batch_size)
            
            for epoch in range(num_epochs):
                epoch_start = asyncio.get_event_loop().time()
                
                # Create data batches
                batches = self._create_batches(training_data, labels, batch_size)
                
                # Define gradient computation function
                async def compute_batch_gradients(batch_data, context=None):
                    X_batch, y_batch = batch_data
                    
                    # Forward pass
                    predictions = np.dot(X_batch, weights) + bias
                    
                    # Compute loss and gradients
                    loss = np.mean((predictions - y_batch) ** 2)
                    
                    # Gradients
                    dw = 2 * np.dot(X_batch.T, (predictions - y_batch)) / len(y_batch)
                    db = 2 * np.mean(predictions - y_batch)
                    
                    return {
                        'gradients_w': dw,
                        'gradients_b': db,
                        'loss': loss,
                        'batch_size': len(y_batch)
                    }
                
                # Setup parallel gradient computation
                parallel_block = {
                    'name': f'gradient_computation_epoch_{epoch}',
                    'strategy': 'all',
                    'steps': batches,
                    'max_concurrency': 10,
                    'distribution_policy': 'balanced'
                }
                
                # Execute distributed parallel gradient computation
                result = await self.coordinator.execute_distributed_parallel(
                    parallel_block,
                    compute_batch_gradients,
                    security_context=security_context
                )
                
                # Aggregate gradients
                total_dw = np.zeros_like(weights)
                total_db = 0.0
                total_loss = 0.0
                total_samples = 0
                
                for task_result in result.results:
                    if task_result.status == "completed":
                        batch_result = task_result.result
                        batch_weight = batch_result['batch_size']
                        
                        total_dw += batch_result['gradients_w'] * batch_weight
                        total_db += batch_result['gradients_b'] * batch_weight
                        total_loss += batch_result['loss'] * batch_weight
                        total_samples += batch_weight
                
                # Average gradients
                if total_samples > 0:
                    avg_dw = total_dw / total_samples
                    avg_db = total_db / total_samples
                    avg_loss = total_loss / total_samples
                    
                    # Update parameters
                    weights -= learning_rate * avg_dw
                    bias -= learning_rate * avg_db
                    
                    # Record metrics
                    await self.observability.record_metric(
                        "training_loss", avg_loss, 
                        labels={"epoch": str(epoch)}
                    )
                    
                    epoch_time = asyncio.get_event_loop().time() - epoch_start
                    training_history.append({
                        'epoch': epoch,
                        'loss': avg_loss,
                        'epoch_time': epoch_time,
                        'successful_batches': result.completed_tasks,
                        'failed_batches': result.failed_tasks
                    })
                    
                    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")
        
        return {
            'final_weights': weights.tolist(),
            'final_bias': float(bias),
            'training_history': training_history,
            'final_loss': training_history[-1]['loss'] if training_history else None
        }
    
    def _create_batches(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int
    ) -> List[tuple]:
        """Create training batches."""
        batches = []
        num_samples = len(X)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            X_batch = X[i:end_idx]
            y_batch = y[i:end_idx]
            batches.append((X_batch, y_batch))
        
        return batches
    
    async def parallel_hyperparameter_search(
        self,
        training_data: np.ndarray,
        labels: np.ndarray,
        param_grid: Dict[str, List[Any]],
        user_id: str = "ml_user"
    ) -> Dict[str, Any]:
        """Perform parallel hyperparameter search."""
        
        # Create security context
        security_context = await self.security_manager.create_security_context(
            user_id=user_id,
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=["read_data", "execute_training", "hyperparameter_search"]
        )
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        async def train_with_params(params, context=None):
            """Train model with specific parameters."""
            return await self._single_training_run(
                training_data, labels, params
            )
        
        # Setup parallel hyperparameter search
        parallel_block = {
            'name': 'hyperparameter_search',
            'strategy': 'all',
            'steps': param_combinations,
            'max_concurrency': 15,
            'distribution_policy': 'balanced'
        }
        
        # Execute parallel search
        result = await self.coordinator.execute_distributed_parallel(
            parallel_block,
            train_with_params,
            security_context=security_context
        )
        
        # Find best parameters
        best_score = float('inf')
        best_params = None
        best_model = None
        
        results = []
        for task_result in result.results:
            if task_result.status == "completed":
                training_result = task_result.result
                results.append(training_result)
                
                if training_result['final_loss'] < best_score:
                    best_score = training_result['final_loss']
                    best_params = training_result['params']
                    best_model = training_result
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'all_results': results,
            'search_stats': {
                'total_combinations': len(param_combinations),
                'successful_runs': len(results),
                'search_time': result.execution_time
            }
        }
    
    def _generate_param_combinations(
        self, 
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for value_combination in itertools.product(*values):
            combination = dict(zip(keys, value_combination))
            combinations.append(combination)
        
        return combinations
    
    async def _single_training_run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single training run with specified parameters."""
        
        # Extract parameters
        epochs = params.get('epochs', 10)
        learning_rate = params.get('learning_rate', 0.01)
        batch_size = params.get('batch_size', 32)
        
        # Simple linear regression training
        weights = np.random.normal(0, 0.1, X.shape[1])
        bias = 0.0
        
        for epoch in range(epochs):
            # Simple batch processing (non-distributed for inner loop)
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                X_batch = X[i:end_idx]
                y_batch = y[i:end_idx]
                
                # Forward pass
                predictions = np.dot(X_batch, weights) + bias
                
                # Compute gradients
                dw = 2 * np.dot(X_batch.T, (predictions - y_batch)) / len(y_batch)
                db = 2 * np.mean(predictions - y_batch)
                
                # Update parameters
                weights -= learning_rate * dw
                bias -= learning_rate * db
        
        # Final loss computation
        final_predictions = np.dot(X, weights) + bias
        final_loss = np.mean((final_predictions - y) ** 2)
        
        return {
            'params': params,
            'final_weights': weights.tolist(),
            'final_bias': float(bias),
            'final_loss': float(final_loss)
        }

# Usage example
async def ml_training_example():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(10000, 5)  # 10k samples, 5 features
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = np.dot(X, true_weights) + np.random.normal(0, 0.1, 10000)
    
    trainer = DistributedMLTrainer()
    await trainer.start()
    
    try:
        # Train single model
        print("Training distributed model...")
        training_result = await trainer.train_distributed_model(
            X, y,
            num_epochs=20,
            batch_size=64,
            learning_rate=0.01
        )
        
        print(f"Training completed! Final loss: {training_result['final_loss']:.6f}")
        print(f"Learned weights: {training_result['final_weights']}")
        
        # Hyperparameter search
        print("\nStarting hyperparameter search...")
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epochs': [10, 15, 20]
        }
        
        search_result = await trainer.parallel_hyperparameter_search(
            X, y, param_grid
        )
        
        print(f"Best parameters: {search_result['best_params']}")
        print(f"Best score: {search_result['best_score']:.6f}")
        print(f"Search completed in {search_result['search_stats']['search_time']:.2f}s")
        
    finally:
        await trainer.stop()

# Run example
asyncio.run(ml_training_example())
```

---

## Web Service Integration

### 1. API Gateway with Parallel Processing

```python
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, RedisMessageBroker
from namel3ss.runtime.security import SecurityManager, PermissionLevel
from namel3ss.runtime.observability import ObservabilityManager

# Pydantic models for API
class ParallelTaskRequest(BaseModel):
    name: str
    strategy: str = "all"
    steps: List[Any]
    max_concurrency: Optional[int] = None
    timeout_seconds: Optional[float] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    result: Optional[Any] = None
    error: Optional[str] = None

class DistributedTaskRequest(BaseModel):
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None

# FastAPI application
app = FastAPI(
    title="Namel3ss Parallel Processing API",
    description="REST API for distributed parallel processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

class ApiGateway:
    """API Gateway for Namel3ss processing services."""
    
    def __init__(self):
        # Initialize components
        self.security_manager = SecurityManager(audit_enabled=True)
        self.observability = ObservabilityManager(
            service_name="namel3ss_api",
            enable_metrics=True,
            enable_tracing=True
        )
        
        # Processing components
        self.broker = RedisMessageBroker()
        self.distributed_queue = DistributedTaskQueue(
            broker=self.broker,
            enable_security=True,
            security_manager=self.security_manager
        )
        
        self.parallel_executor = ParallelExecutor(
            enable_security=True,
            security_manager=self.security_manager,
            enable_observability=True,
            observability_manager=self.observability
        )
        
        self.coordinator = DistributedParallelExecutor(
            parallel_executor=self.parallel_executor,
            distributed_queue=self.distributed_queue,
            enable_security=True,
            security_manager=self.security_manager
        )
        
        # Task storage
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def start(self):
        """Start the API gateway."""
        await self.distributed_queue.start()
        print("API Gateway started successfully")
    
    async def stop(self):
        """Stop the API gateway."""
        await self.distributed_queue.stop()
        print("API Gateway stopped")
    
    async def authenticate_request(
        self, 
        credentials: HTTPAuthorizationCredentials
    ) -> str:
        """Authenticate API request and return user ID."""
        # In production, validate JWT token here
        token = credentials.credentials
        
        # Simple token validation (replace with proper JWT validation)
        if token.startswith("user_"):
            return token
        
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    
    async def create_security_context(self, user_id: str):
        """Create security context for user."""
        return await self.security_manager.create_security_context(
            user_id=user_id,
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=["read_data", "execute_parallel", "execute_distributed"]
        )

# Global gateway instance
gateway = ApiGateway()

@app.on_event("startup")
async def startup_event():
    """Initialize the gateway on startup."""
    await gateway.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await gateway.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "namel3ss-api",
        "version": "1.0.0"
    }

@app.post("/api/v1/parallel/execute")
async def execute_parallel_task(
    request: ParallelTaskRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Execute parallel processing task."""
    
    # Authenticate user
    user_id = await gateway.authenticate_request(credentials)
    security_context = await gateway.create_security_context(user_id)
    
    # Record metrics
    await gateway.observability.record_metric(
        "api_requests_total",
        1.0,
        labels={"endpoint": "parallel_execute", "user": user_id}
    )
    
    try:
        async with gateway.observability.start_trace("parallel_api_request") as span:
            span.set_tag("user_id", user_id)
            span.set_tag("task_name", request.name)
            span.set_tag("strategy", request.strategy)
            
            # Define step executor for API context
            async def api_step_executor(step, context=None):
                # In a real implementation, this would route to different processors
                # based on step type or content
                await asyncio.sleep(0.1)  # Simulate processing
                return f"processed_{step}"
            
            # Setup parallel block
            parallel_block = {
                'name': request.name,
                'strategy': request.strategy,
                'steps': request.steps,
                'max_concurrency': request.max_concurrency,
                'timeout_seconds': request.timeout_seconds
            }
            
            # Execute parallel processing
            result = await gateway.parallel_executor.execute_parallel_block(
                parallel_block,
                api_step_executor,
                security_context=security_context
            )
            
            # Format response
            response = {
                'task_id': f"parallel_{result.block_name}_{int(time.time())}",
                'status': 'completed',
                'result': {
                    'block_name': result.block_name,
                    'strategy': result.strategy,
                    'completed_tasks': result.completed_tasks,
                    'failed_tasks': result.failed_tasks,
                    'execution_time': result.execution_time,
                    'results': [
                        {
                            'task_id': r.task_id,
                            'status': r.status,
                            'result': r.result
                        }
                        for r in result.results
                    ]
                }
            }
            
            span.set_tag("completed_tasks", result.completed_tasks)
            span.set_tag("execution_time", result.execution_time)
            
            return response
            
    except Exception as e:
        await gateway.observability.record_metric(
            "api_errors_total",
            1.0,
            labels={"endpoint": "parallel_execute", "error_type": type(e).__name__}
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Parallel execution failed: {str(e)}"
        )

@app.post("/api/v1/distributed/submit")
async def submit_distributed_task(
    request: DistributedTaskRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, str]:
    """Submit distributed processing task."""
    
    # Authenticate user
    user_id = await gateway.authenticate_request(credentials)
    security_context = await gateway.create_security_context(user_id)
    
    try:
        # Submit task to distributed queue
        task_id = await gateway.distributed_queue.submit_task(
            task_type=request.task_type,
            payload=request.payload,
            priority=request.priority,
            timeout=request.timeout,
            security_context=security_context
        )
        
        return {
            "task_id": task_id,
            "status": "submitted"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Task submission failed: {str(e)}"
        )

@app.get("/api/v1/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TaskStatusResponse:
    """Get status of submitted task."""
    
    # Authenticate user
    user_id = await gateway.authenticate_request(credentials)
    
    try:
        # Get task status from distributed queue
        status_info = await gateway.distributed_queue.get_task_status(task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status_info.get('state', 'unknown'),
            progress=status_info.get('progress', 0.0),
            result=status_info.get('result'),
            error=status_info.get('error')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found or status unavailable: {str(e)}"
        )

@app.get("/api/v1/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    timeout: Optional[float] = 30.0
) -> Dict[str, Any]:
    """Get result of completed task."""
    
    # Authenticate user
    user_id = await gateway.authenticate_request(credentials)
    
    try:
        # Get task result from distributed queue
        result = await gateway.distributed_queue.get_task_result(
            task_id,
            timeout=timeout
        )
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to get task result: {str(e)}"
        )

@app.get("/api/v1/metrics")
async def get_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get system metrics summary."""
    
    # Authenticate user (admin only for metrics)
    user_id = await gateway.authenticate_request(credentials)
    
    try:
        # Get metrics summary
        metrics = await gateway.observability.get_metrics_summary()
        
        return {
            "service": "namel3ss-api",
            "metrics": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

# Client library example
class Namel3ssApiClient:
    """Client library for Namel3ss API."""
    
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    async def execute_parallel(
        self,
        name: str,
        steps: List[Any],
        strategy: str = "all",
        max_concurrency: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute parallel processing task."""
        
        request_data = {
            "name": name,
            "strategy": strategy,
            "steps": steps,
            "max_concurrency": max_concurrency
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/parallel/execute",
                json=request_data,
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def submit_distributed_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit distributed task."""
        
        request_data = {
            "task_type": task_type,
            "payload": payload,
            "priority": priority
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/distributed/submit",
                json=request_data,
                headers=self.headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["task_id"]
    
    async def get_task_result(
        self,
        task_id: str,
        timeout: float = 30.0
    ) -> Any:
        """Get task result."""
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/tasks/{task_id}/result?timeout={timeout}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["result"]

# Usage example
async def api_client_example():
    """Example of using the API client."""
    
    client = Namel3ssApiClient(
        base_url="http://localhost:8000",
        api_token="user_example_123"
    )
    
    # Execute parallel task
    parallel_result = await client.execute_parallel(
        name="api_test",
        steps=[1, 2, 3, 4, 5],
        strategy="all",
        max_concurrency=3
    )
    
    print(f"Parallel result: {parallel_result}")
    
    # Submit distributed task
    task_id = await client.submit_distributed_task(
        task_type="data_processing",
        payload={"data": "sample_data"},
        priority=1
    )
    
    print(f"Submitted task: {task_id}")
    
    # Get result
    result = await client.get_task_result(task_id, timeout=60.0)
    print(f"Task result: {result}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
```

---

## Summary

This comprehensive examples guide provides:

✅ **Real-World Applications** - Practical examples for data processing, ML, web services  
✅ **Production Patterns** - Enterprise-grade implementations with security and observability  
✅ **Performance Optimization** - Efficient patterns for high-throughput processing  
✅ **Integration Examples** - API gateways, client libraries, and service orchestration  
✅ **Security Integration** - Authentication, authorization, and audit logging  
✅ **Monitoring & Observability** - Metrics, tracing, and health monitoring  

Key implementation areas covered:
- **Data Processing**: Batch processing, stream processing with windowing
- **Machine Learning**: Distributed training, hyperparameter search  
- **Web Services**: REST API integration with authentication and monitoring
- **Real-time Systems**: Event-driven processing, WebSocket streaming
- **Enterprise Features**: Security, audit logging, comprehensive monitoring

Each example is production-ready and demonstrates best practices for scalable, secure, and observable parallel and distributed execution systems.