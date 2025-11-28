# Extensions Guide: Custom Tools and Processing

**Purpose**: Learn how to extend Namel3ss with custom Python functions for file processing, external API calls, scheduling, and other advanced operations not available in the core DSL.

---

## Key Concept: Core vs Extensions

Namel3ss has a **focused core** with clear extension points:

**Core DSL** (Built-in):
- UI components (show text, show form, show table)
- Dataset operations (filter, sort, aggregate)
- Prompts and AI chains
- Memory and context management

**Extensions** (Via Python Tools):
- File upload and processing
- Custom computations
- External API integration
- Background jobs and scheduling
- Complex data transformations

---

## Tool Pattern: Python Functions

The `tool` declaration lets you integrate custom Python functions into your Namel3ss application.

### Basic Tool Definition

**Python file** (`tools/file_processor.py`):
```python
def process_pdf(file_path: str) -> dict:
    """Extract text and metadata from a PDF file."""
    import PyPDF2
    
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        return {
            "page_count": len(reader.pages),
            "text": "".join(page.extract_text() for page in reader.pages),
            "metadata": reader.metadata
        }
```

**Namel3ss declaration**:
```namel3ss
tool process_pdf:
    module: "tools.file_processor"
    function: "process_pdf"
    inputs:
        file_path: text
    outputs:
        page_count: integer
        text: text
        metadata: json
```

### Using Tools in Chains

```namel3ss
define chain "ProcessDocument":
    inputs:
        file_id: text
    
    steps:
        - step extract:
            call tool process_pdf:
                file_path: file_id
            store result as doc_data
        
        - step analyze:
            run prompt AnalyzeText:
                content: doc_data.text
            store result as analysis
        
        - step save:
            update Jobs:
                where: id == file_id
                set:
                    status: "completed"
                    page_count: doc_data.page_count
                    analysis: analysis.summary
```

---

## File Processing

### File Upload Tool

**Python** (`tools/file_handler.py`):
```python
import os
import shutil
from pathlib import Path

def save_uploaded_file(file_data: bytes, filename: str, upload_dir: str = "uploads") -> dict:
    """Save uploaded file to disk."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    
    file_path = os.path.join(upload_dir, filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_data)
    
    file_size = os.path.getsize(file_path)
    
    return {
        "file_path": file_path,
        "file_size": file_size,
        "saved_at": datetime.now().isoformat()
    }

def validate_file(file_path: str, max_size_mb: int = 10, allowed_types: list = None) -> dict:
    """Validate file size and type."""
    allowed_types = allowed_types or ['.pdf', '.docx', '.txt']
    
    file_size = os.path.getsize(file_path)
    file_ext = Path(file_path).suffix.lower()
    
    errors = []
    
    if file_size > max_size_mb * 1024 * 1024:
        errors.append(f"File too large: {file_size / (1024*1024):.2f}MB (max {max_size_mb}MB)")
    
    if file_ext not in allowed_types:
        errors.append(f"File type {file_ext} not allowed (allowed: {', '.join(allowed_types)})")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "file_size": file_size,
        "file_type": file_ext
    }
```

**Namel3ss**:
```namel3ss
tool save_uploaded_file:
    module: "tools.file_handler"
    function: "save_uploaded_file"
    inputs:
        file_data: bytes
        filename: text
    outputs:
        file_path: text
        file_size: integer

tool validate_file:
    module: "tools.file_handler"
    function: "validate_file"
    inputs:
        file_path: text
        max_size_mb: integer
        allowed_types: list<text>
    outputs:
        valid: boolean
        errors: list<text>

page "UploadDocument" at "/upload":
    show form "UploadForm":
        fields:
            document:
                type: "file"
                accept: ".pdf,.docx,.txt"
                required: true
            description:
                type: "textarea"
        
        on submit:
            # Save file
            call tool save_uploaded_file:
                file_data: form.document
                filename: form.document.name
            store result as saved_file
            
            # Validate
            call tool validate_file:
                file_path: saved_file.file_path
                max_size_mb: 10
                allowed_types: [".pdf", ".docx", ".txt"]
            store result as validation
            
            if validation.valid:
                # Create job record
                insert into FileProcessingJob:
                    filename: form.document.name
                    file_path: saved_file.file_path
                    description: form.description
                    status: "pending"
                    created_at: now()
                
                go to page "JobList"
                show toast "File uploaded successfully"
            else:
                show toast "Validation failed: {{join(validation.errors, ', ')}}"
```

### File Processing Examples

**Image Processing** (`tools/image_processor.py`):
```python
from PIL import Image

def resize_image(input_path: str, output_path: str, max_width: int = 800, max_height: int = 600) -> dict:
    """Resize image maintaining aspect ratio."""
    img = Image.open(input_path)
    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    img.save(output_path, optimize=True, quality=85)
    
    return {
        "width": img.width,
        "height": img.height,
        "output_path": output_path
    }

def extract_metadata(image_path: str) -> dict:
    """Extract EXIF metadata from image."""
    img = Image.open(image_path)
    exif = img.getexif()
    
    return {
        "format": img.format,
        "mode": img.mode,
        "size": {"width": img.width, "height": img.height},
        "exif": {str(k): str(v) for k, v in exif.items()} if exif else {}
    }
```

**CSV Processing** (`tools/csv_processor.py`):
```python
import csv
import pandas as pd

def parse_csv(file_path: str, max_rows: int = 1000) -> dict:
    """Parse CSV file and return data."""
    df = pd.read_csv(file_path, nrows=max_rows)
    
    return {
        "columns": list(df.columns),
        "row_count": len(df),
        "data": df.to_dict(orient='records'),
        "summary": df.describe().to_dict()
    }

def validate_csv_schema(file_path: str, required_columns: list) -> dict:
    """Validate CSV has required columns."""
    df = pd.read_csv(file_path, nrows=1)
    columns = set(df.columns)
    required = set(required_columns)
    
    missing = required - columns
    
    return {
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "found_columns": list(columns)
    }
```

---

## External API Integration

### HTTP Connector (Already Built-in)

Namel3ss already has `connector` for HTTP calls:

```namel3ss
connector WeatherAPI:
    base_url: "https://api.weather.com"
    headers:
        Authorization: "Bearer {{env:WEATHER_API_KEY}}"

page "Weather" at "/weather":
    show form "LocationForm":
        fields:
            city:
                type: "text"
                placeholder: "Enter city name"
        
        on submit:
            ask connector WeatherAPI:
                method: "GET"
                path: "/v1/current"
                params:
                    q: form.city
            store response as weather_data
            
            show card:
                show text "Temperature: {{weather_data.temp}}°C"
                show text "Conditions: {{weather_data.condition}}"
```

### Custom API Tool

For more complex API interactions, use Python tools:

**Python** (`tools/api_client.py`):
```python
import requests

def call_external_api(
    url: str,
    method: str = "GET",
    headers: dict = None,
    data: dict = None,
    timeout: int = 30
) -> dict:
    """Make HTTP request to external API."""
    response = requests.request(
        method=method,
        url=url,
        headers=headers or {},
        json=data,
        timeout=timeout
    )
    
    return {
        "status_code": response.status_code,
        "success": response.ok,
        "data": response.json() if response.ok else None,
        "error": response.text if not response.ok else None
    }

def batch_api_call(urls: list, method: str = "GET") -> dict:
    """Make multiple API calls in parallel."""
    import concurrent.futures
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_external_api, url, method) for url in urls]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return {
        "total": len(urls),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }
```

---

## Scheduling and Background Jobs

Namel3ss doesn't have built-in scheduling, but you can integrate with external schedulers.

### Pattern 1: Celery Integration

**Python** (`tasks.py`):
```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def process_pending_jobs():
    """Background task to process pending jobs."""
    # Query database for pending jobs
    from your_app.database import get_pending_jobs, update_job_status
    
    jobs = get_pending_jobs()
    
    for job in jobs:
        try:
            # Process the job
            result = process_file(job['file_path'])
            
            # Update status
            update_job_status(job['id'], 'completed', result)
        except Exception as e:
            update_job_status(job['id'], 'failed', error=str(e))
```

**Trigger from Namel3ss**:
```namel3ss
tool trigger_background_job:
    module: "tasks"
    function: "process_pending_jobs"
    async: true

page "ProcessJobs" at "/jobs/process":
    show button "Process All Pending":
        on click:
            call tool trigger_background_job
            show toast "Background processing started"
```

### Pattern 2: Cron Jobs

**Cron** (`crontab -e`):
```bash
# Run every 5 minutes
*/5 * * * * cd /app && python tasks/process_jobs.py

# Run daily at 2 AM
0 2 * * * cd /app && python tasks/cleanup_old_files.py

# Run every hour
0 * * * * cd /app && python tasks/sync_data.py
```

**Python** (`tasks/process_jobs.py`):
```python
#!/usr/bin/env python3
import sys
sys.path.append('/app')

from your_app.processing import process_pending_jobs

if __name__ == '__main__':
    process_pending_jobs()
```

### Pattern 3: APScheduler

**Python** (`scheduler.py`):
```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

scheduler = BackgroundScheduler()

def process_jobs():
    """Process pending jobs."""
    print(f"Processing jobs at {datetime.now()}")
    # Your processing logic here

def cleanup_old_files():
    """Clean up old files."""
    import os
    from datetime import timedelta
    
    cutoff = datetime.now() - timedelta(days=30)
    # Delete files older than 30 days

# Schedule jobs
scheduler.add_job(process_jobs, 'interval', minutes=5)
scheduler.add_job(cleanup_old_files, 'cron', hour=2)

scheduler.start()
```

---

## Monitoring and Job Status

### Job Status Dashboard Pattern

```namel3ss
dataset "JobStatistics" from postgres "jobs":
    schema:
        status: text
        count: integer
        avg_duration: float
        failed_count: integer
    group by: status
    aggregate:
        count: count(id)
        avg_duration: avg(completed_at - created_at)
        failed_count: sum(if(status == "failed", 1, 0))

dataset "RecentJobs" from postgres "jobs":
    schema:
        id: uuid
        filename: text
        status: text
        created_at: timestamp
        completed_at: timestamp
        error_message: text
    order by: created_at desc
    limit: 50

page "JobsDashboard" at "/jobs":
    show card "Overview":
        for stat in JobStatistics:
            show text "{{stat.status}}: {{stat.count}}"
            if stat.avg_duration:
                show text "Avg time: {{format_duration(stat.avg_duration)}}"
    
    show table "Recent Jobs":
        data: RecentJobs
        columns:
            - name: "Filename"
              value: filename
            - name: "Status"
              value: status
            - name: "Created"
              value: format_timestamp(created_at, 'relative')
            - name: "Duration"
              value: format_duration(completed_at - created_at)
        
        row_actions:
            - if row.status == "failed":
                show button "Retry":
                    on click:
                        update Jobs:
                            where: id == row.id
                            set:
                                status: "pending"
                                error_message: null
                        refresh data
```

---

## Error Handling in Tools

### Tool with Error Handling

**Python**:
```python
def safe_process_file(file_path: str) -> dict:
    """Process file with error handling."""
    try:
        result = process_file(file_path)
        return {
            "success": True,
            "data": result,
            "error": None
        }
    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": "File not found"
        }
    except PermissionError:
        return {
            "success": False,
            "data": None,
            "error": "Permission denied"
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }
```

**Usage**:
```namel3ss
define chain "ProcessWithErrorHandling":
    inputs:
        file_id: text
    
    steps:
        - step process:
            call tool safe_process_file:
                file_path: file_id
            store result as result
        
        - step handle_result:
            if result.success:
                update Jobs:
                    where: id == file_id
                    set:
                        status: "completed"
                        data: result.data
            else:
                update Jobs:
                    where: id == file_id
                    set:
                        status: "failed"
                        error_message: result.error
                
                show toast "Processing failed: {{result.error}}"
```

---

## Best Practices

### 1. Keep Tools Focused

✅ **Good**: Single responsibility
```python
def extract_text(file_path: str) -> str:
    """Extract text from file."""
    # Single, clear purpose
```

❌ **Bad**: Too many responsibilities
```python
def process_file(file_path: str) -> dict:
    # Extracts, analyzes, sends email, updates database...
    # Too many things in one function
```

### 2. Return Structured Data

✅ **Good**: Clear return structure
```python
def process_file(file_path: str) -> dict:
    return {
        "success": True,
        "page_count": 10,
        "text": "...",
        "metadata": {...}
    }
```

❌ **Bad**: Unclear return value
```python
def process_file(file_path: str):
    return (10, "...", {...})  # What does each value mean?
```

### 3. Handle Errors Gracefully

✅ **Good**: Return error information
```python
def process_file(file_path: str) -> dict:
    try:
        result = do_processing(file_path)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 4. Use Type Hints

✅ **Good**: Clear types
```python
def process_csv(file_path: str, max_rows: int = 1000) -> dict:
    ...
```

---

## Complete Example: File Processing App

See `examples/processing_tools/` for a complete working example including:

- File upload with validation
- Background processing with status tracking
- Dashboard with job statistics
- Error handling and retry logic
- Custom Python tools for file operations

---

## Related Documentation

- [DATA_MODELS_GUIDE.md](./DATA_MODELS_GUIDE.md) - Dataset and frame usage
- [STANDARD_LIBRARY.md](./STANDARD_LIBRARY.md) - Built-in functions
- [API_REFERENCE.md](./API_REFERENCE.md) - Connector documentation
- [PROCESSING_AUTOMATION_GUIDE.md](./PROCESSING_AUTOMATION_GUIDE.md) - Complete guide

---

**Last Updated**: November 28, 2025
