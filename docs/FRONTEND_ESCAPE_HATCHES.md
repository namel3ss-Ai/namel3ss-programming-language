# Frontend Escape Hatches: Building UI Tools for N3 Workflows

## Overview

**Audience:** Senior frontend engineers, full-stack engineers, platform engineers building production SaaS applications with N3.

**Purpose:** This document explains how to build "escape hatches" that allow N3 workflows to interact with frontend UI, events, and browser APIs when pure backend execution isn't sufficient. Escape hatches enable workflows to request user input, display modals, upload files, and access browser-only capabilities while maintaining workflow orchestration in N3.

**Context:** N3 workflows typically run in backend services (FastAPI, Node.js), but production applications often need workflows to:
- Request user approvals mid-execution
- Display real-time progress or notifications
- Access browser APIs (camera, geolocation, clipboard)
- Navigate to different pages based on workflow results
- Trigger UI updates without polling

This document covers **how to architect, implement, and operate these bidirectional integrations** at production scale.

---

## When to Use Escape Hatches

### ✅ Use Escape Hatches When:

1. **Human-in-the-loop workflows**: Payment approvals, content moderation, risk reviews
2. **File upload workflows**: User must select/upload files during workflow execution
3. **Real-time feedback**: Long-running workflows that need progress bars, status updates
4. **Browser-only APIs**: Workflows need camera access, geolocation, notifications
5. **Dynamic navigation**: Workflow result determines next page (success → dashboard, error → retry form)
6. **Conditional UI updates**: Show/hide elements based on workflow state without full page reload

### ❌ Do NOT Use Escape Hatches When:

1. **Pure backend automation**: No user interaction needed (scheduled jobs, background processing)
2. **Simple request-response**: Standard API calls work fine (GET user profile, POST form data)
3. **Synchronous operations**: Fast workflows that complete before user can react (<200ms)
4. **Pre-workflow input collection**: Gather inputs in UI first, then execute workflow with all data

---

## Architecture Patterns

### Pattern 1: Tool-Based Escape Hatches (Recommended)

**How it works:** Frontend registers "tools" (e.g., `ui.showModal`, `ui.uploadFile`) that N3 workflows can invoke. Backend sends tool call request → frontend executes → frontend sends result back.

**Best for:** Hybrid mode (remote backend + frontend tools)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tool-Based Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐         WebSocket          ┌───────────────┐  │
│  │  Frontend   │◄────────────────────────────│  N3 Backend   │  │
│  │             │                             │               │  │
│  │  Registers: │  1. Register tools          │  Executes:    │  │
│  │  - ui.modal │  ──────────────────────────►│  - workflows  │  │
│  │  - ui.upload│                             │  - tool calls │  │
│  │  - ui.notify│  2. tool_call_requested     │               │  │
│  │             │◄────────────────────────────│               │  │
│  │             │     {tool: "ui.modal"}      │               │  │
│  │             │                             │               │  │
│  │  Executes:  │  3. tool_call_response      │               │  │
│  │  - show UI  │  ──────────────────────────►│  Continues    │  │
│  │  - wait for │     {result: "approved"}    │  execution    │  │
│  │    user     │                             │               │  │
│  └─────────────┘                             └───────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Pattern 2: Event-Based Escape Hatches

**How it works:** N3 workflow emits events (e.g., `step_completed`, `progress_update`), frontend listens and updates UI reactively.

**Best for:** One-way updates (backend → frontend), no user interaction needed

```
┌─────────────────────────────────────────────────────────────────┐
│                   Event-Based Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐         SSE/WebSocket       ┌───────────────┐ │
│  │  Frontend   │◄────────────────────────────│  N3 Backend   │ │
│  │             │                             │               │ │
│  │  Listens:   │  workflow_started           │  Emits:       │ │
│  │  - events   │◄────────────────────────────│  - events     │ │
│  │             │                             │               │ │
│  │  Updates:   │  step_completed (1/5)       │  During:      │ │
│  │  - progress │◄────────────────────────────│  - execution  │ │
│  │  - status   │                             │               │ │
│  │  - UI state │  progress_update (40%)      │               │ │
│  │             │◄────────────────────────────│               │ │
│  │             │                             │               │ │
│  │             │  workflow_completed         │               │ │
│  │             │◄────────────────────────────│               │ │
│  └─────────────┘                             └───────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Pattern 3: Callback URLs

**How it works:** Workflow generates unique callback URL, sends to user (email, SMS), user clicks link to continue workflow.

**Best for:** Long-running workflows where user might not be online (async approvals, email confirmations)

```
┌─────────────────────────────────────────────────────────────────┐
│                  Callback URL Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                             ┌───────────────┐ │
│  │  Frontend   │                             │  N3 Backend   │ │
│  │             │  1. POST /execute           │               │ │
│  │             │ ──────────────────────────► │  Starts       │ │
│  │             │                             │  workflow     │ │
│  │             │  2. Execution ID            │               │ │
│  │             │◄────────────────────────────│               │ │
│  │             │                             │               │ │
│  │             │                             │  3. Sends     │ │
│  │             │                             │     email     │ │
│  │             │                             │     with URL  │ │
│  │             │                             │               │ │
│  │  User       │  4. Click link in email     │               │ │
│  │  clicks     │     /approve?token=xyz      │               │ │
│  │  ──────────►│  ──────────────────────────►│  Resumes      │ │
│  │             │                             │  workflow     │ │
│  │             │  5. Success page            │               │ │
│  │             │◄────────────────────────────│               │ │
│  └─────────────┘                             └───────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Section 1: UI Tool Escape Hatches

### Registering UI Tools

Frontend registers tools that backend can invoke during workflow execution.

**Tool Registration (Frontend):**

```typescript
// lib/n3-tools.ts
import { WebSocket } from 'ws';

export interface UITool {
  id: string;
  schema: Record<string, any>;
  handler: (args: any) => Promise<any>;
}

export class N3ToolRegistry {
  private ws: WebSocket;
  private tools: Map<string, UITool> = new Map();
  private pendingCalls: Map<string, { resolve: Function; reject: Function }> = new Map();

  constructor(wsUrl: string, authToken: string) {
    this.ws = new WebSocket(`${wsUrl}?token=${authToken}`);
    this.setupWebSocket();
  }

  private setupWebSocket() {
    this.ws.onopen = () => {
      console.log('N3 WebSocket connected');
      this.registerAllTools();
    };

    this.ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === 'tool_call_requested') {
        await this.handleToolCall(msg);
      }
    };

    this.ws.onerror = (error) => {
      console.error('N3 WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.warn('N3 WebSocket closed, reconnecting...');
      setTimeout(() => this.reconnect(), 2000);
    };
  }

  registerTool(tool: UITool) {
    this.tools.set(tool.id, tool);
  }

  private registerAllTools() {
    const toolDefinitions = Array.from(this.tools.values()).map((tool) => ({
      id: tool.id,
      schema: tool.schema,
    }));

    this.ws.send(
      JSON.stringify({
        type: 'register_tools',
        tools: toolDefinitions,
      })
    );

    console.log(`Registered ${toolDefinitions.length} UI tools`);
  }

  private async handleToolCall(msg: {
    call_id: string;
    tool: string;
    args: any;
    execution_id: string;
  }) {
    const { call_id, tool, args, execution_id } = msg;

    console.log(`Tool call requested: ${tool}`, args);

    try {
      const toolHandler = this.tools.get(tool);
      if (!toolHandler) {
        throw new Error(`Tool not registered: ${tool}`);
      }

      // Execute tool handler
      const result = await toolHandler.handler(args);

      // Send result back to backend
      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result,
          error: null,
        })
      );

      console.log(`Tool call completed: ${tool}`, result);
    } catch (error) {
      console.error(`Tool call failed: ${tool}`, error);

      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result: null,
          error: error.message,
        })
      );
    }
  }

  private reconnect() {
    this.ws = new WebSocket(this.ws.url);
    this.setupWebSocket();
  }

  close() {
    this.ws.close();
  }
}
```

---

### Tool 1: Modal Dialogs

**Use Case:** Workflow needs to show confirmation dialog, alert, or multi-choice question.

**N3 Workflow:**
```python
# approval_workflow.ai
workflow PaymentApproval {
  steps: [
    {
      "id": "show_confirmation",
      "tool": "ui.showModal",
      "args": {
        "title": "Approve Payment",
        "message": "Approve payment of $5,000 to Acme Corp?",
        "actions": ["Approve", "Reject", "Request More Info"],
        "variant": "warning",
        "timeout_ms": 300000  # 5 minutes
      }
    },
    {
      "id": "process_payment",
      "tool": "payments.process",
      "args": {"amount": 5000},
      "condition": "steps.show_confirmation.result == 'Approve'"
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/modal-tool.tsx
import { useState, useEffect } from 'react';
import { UITool } from '@/lib/n3-tools';

interface ModalArgs {
  title: string;
  message: string;
  actions: string[];
  variant?: 'info' | 'warning' | 'danger';
  timeout_ms?: number;
}

let modalResolver: ((result: string) => void) | null = null;

export const useModalTool = (): UITool => {
  return {
    id: 'ui.showModal',
    schema: {
      title: { type: 'string' },
      message: { type: 'string' },
      actions: { type: 'array', items: { type: 'string' } },
      variant: { type: 'string', enum: ['info', 'warning', 'danger'] },
      timeout_ms: { type: 'number' },
    },
    handler: async (args: ModalArgs) => {
      return new Promise((resolve, reject) => {
        // Store resolver for modal component to call
        modalResolver = resolve;

        // Emit custom event to show modal
        window.dispatchEvent(
          new CustomEvent('n3:showModal', { detail: args })
        );

        // Handle timeout
        if (args.timeout_ms) {
          setTimeout(() => {
            if (modalResolver) {
              reject(new Error('Modal timeout'));
              modalResolver = null;
            }
          }, args.timeout_ms);
        }
      });
    },
  };
};

// Modal component that listens for n3:showModal events
export function N3ModalContainer() {
  const [isOpen, setIsOpen] = useState(false);
  const [modalData, setModalData] = useState<ModalArgs | null>(null);

  useEffect(() => {
    const handleShowModal = (event: CustomEvent<ModalArgs>) => {
      setModalData(event.detail);
      setIsOpen(true);
    };

    window.addEventListener('n3:showModal', handleShowModal as any);

    return () => {
      window.removeEventListener('n3:showModal', handleShowModal as any);
    };
  }, []);

  const handleAction = (action: string) => {
    if (modalResolver) {
      modalResolver(action);
      modalResolver = null;
    }
    setIsOpen(false);
    setModalData(null);
  };

  if (!isOpen || !modalData) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-semibold mb-2">{modalData.title}</h2>
        <p className="text-gray-700 mb-6">{modalData.message}</p>

        <div className="flex gap-3 justify-end">
          {modalData.actions.map((action) => (
            <button
              key={action}
              onClick={() => handleAction(action)}
              className={`px-4 py-2 rounded ${
                modalData.variant === 'danger' && action === modalData.actions[0]
                  ? 'bg-red-600 text-white'
                  : 'bg-blue-600 text-white'
              }`}
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**Usage in App:**

```typescript
// app/layout.tsx
import { N3ModalContainer } from '@/components/ui/modal-tool';
import { N3ToolRegistry } from '@/lib/n3-tools';
import { useModalTool } from '@/components/ui/modal-tool';

export default function RootLayout({ children }) {
  useEffect(() => {
    const registry = new N3ToolRegistry(
      'wss://api.example.com/ws',
      localStorage.getItem('auth_token')
    );

    // Register modal tool
    const modalTool = useModalTool();
    registry.registerTool(modalTool);

    return () => registry.close();
  }, []);

  return (
    <html>
      <body>
        {children}
        <N3ModalContainer />
      </body>
    </html>
  );
}
```

---

### Tool 2: File Upload

**Use Case:** Workflow needs user to upload file (document verification, image processing).

**N3 Workflow:**
```python
# document_verification.ai
workflow DocumentVerification {
  steps: [
    {
      "id": "request_id_upload",
      "tool": "ui.uploadFile",
      "args": {
        "title": "Upload Government ID",
        "accept": ".pdf,.jpg,.png",
        "maxSize": 10485760,  # 10 MB
        "instructions": "Please upload a clear photo of your government-issued ID"
      }
    },
    {
      "id": "verify_document",
      "tool": "ocr.extract_and_verify",
      "args": {
        "file_url": steps.request_id_upload.file_url,
        "file_type": steps.request_id_upload.mime_type
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/file-upload-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { useState, useRef } from 'react';

interface FileUploadArgs {
  title: string;
  accept: string;
  maxSize: number;
  instructions?: string;
}

interface FileUploadResult {
  file_url: string;
  file_name: string;
  mime_type: string;
  size: number;
}

let fileUploadResolver: ((result: FileUploadResult) => void) | null = null;
let fileUploadRejecter: ((error: Error) => void) | null = null;

export const useFileUploadTool = (): UITool => {
  return {
    id: 'ui.uploadFile',
    schema: {
      title: { type: 'string' },
      accept: { type: 'string' },
      maxSize: { type: 'number' },
      instructions: { type: 'string' },
    },
    handler: async (args: FileUploadArgs) => {
      return new Promise((resolve, reject) => {
        fileUploadResolver = resolve;
        fileUploadRejecter = reject;

        window.dispatchEvent(
          new CustomEvent('n3:uploadFile', { detail: args })
        );
      });
    },
  };
};

export function N3FileUploadContainer() {
  const [isOpen, setIsOpen] = useState(false);
  const [uploadData, setUploadData] = useState<FileUploadArgs | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleUploadRequest = (event: CustomEvent<FileUploadArgs>) => {
      setUploadData(event.detail);
      setIsOpen(true);
      setError(null);
    };

    window.addEventListener('n3:uploadFile', handleUploadRequest as any);

    return () => {
      window.removeEventListener('n3:uploadFile', handleUploadRequest as any);
    };
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !uploadData) return;

    // Validate file size
    if (file.size > uploadData.maxSize) {
      const maxSizeMB = (uploadData.maxSize / 1048576).toFixed(1);
      setError(`File too large. Maximum size: ${maxSizeMB} MB`);
      return;
    }

    setUploading(true);
    setError(null);

    try {
      // Upload to storage (S3, Cloudinary, etc.)
      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const { file_url } = await uploadResponse.json();

      // Resolve with file metadata
      if (fileUploadResolver) {
        fileUploadResolver({
          file_url,
          file_name: file.name,
          mime_type: file.type,
          size: file.size,
        });
        fileUploadResolver = null;
      }

      // Close modal
      setIsOpen(false);
      setUploadData(null);
    } catch (err) {
      setError('Upload failed. Please try again.');
      if (fileUploadRejecter) {
        fileUploadRejecter(new Error('Upload failed'));
        fileUploadRejecter = null;
      }
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    if (fileUploadRejecter) {
      fileUploadRejecter(new Error('Upload cancelled'));
      fileUploadRejecter = null;
    }
    setIsOpen(false);
    setUploadData(null);
  };

  if (!isOpen || !uploadData) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-xl font-semibold mb-2">{uploadData.title}</h2>
        
        {uploadData.instructions && (
          <p className="text-gray-600 text-sm mb-4">{uploadData.instructions}</p>
        )}

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-4">
          <input
            ref={fileInputRef}
            type="file"
            accept={uploadData.accept}
            onChange={handleFileSelect}
            className="hidden"
            disabled={uploading}
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {uploading ? 'Uploading...' : 'Select File'}
          </button>
          
          <p className="text-gray-500 text-sm mt-2">
            Accepted: {uploadData.accept}
          </p>
          <p className="text-gray-500 text-sm">
            Max size: {(uploadData.maxSize / 1048576).toFixed(1)} MB
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={handleCancel}
            disabled={uploading}
            className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

### Tool 3: Toast Notifications

**Use Case:** Workflow wants to show non-blocking notifications (success, error, info).

**N3 Workflow:**
```python
# background_sync.ai
workflow BackgroundSync {
  steps: [
    {
      "id": "notify_start",
      "tool": "ui.notify",
      "args": {
        "message": "Sync started in background",
        "type": "info",
        "duration": 3000
      }
    },
    {
      "id": "sync_data",
      "tool": "sync.execute",
      "args": {"source": "salesforce"}
    },
    {
      "id": "notify_complete",
      "tool": "ui.notify",
      "args": {
        "message": "Sync completed successfully",
        "type": "success",
        "duration": 5000
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/notification-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { toast } from 'sonner';  // Using sonner toast library

interface NotifyArgs {
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  duration?: number;
}

export const useNotificationTool = (): UITool => {
  return {
    id: 'ui.notify',
    schema: {
      message: { type: 'string' },
      type: { type: 'string', enum: ['info', 'success', 'warning', 'error'] },
      duration: { type: 'number' },
    },
    handler: async (args: NotifyArgs) => {
      // Show toast notification
      const options = {
        duration: args.duration || 4000,
      };

      switch (args.type) {
        case 'success':
          toast.success(args.message, options);
          break;
        case 'error':
          toast.error(args.message, options);
          break;
        case 'warning':
          toast.warning(args.message, options);
          break;
        case 'info':
        default:
          toast.info(args.message, options);
      }

      // Return immediately (non-blocking)
      return { shown: true };
    },
  };
};
```

---

## Section 2: Event-Based Escape Hatches

Event-based escape hatches enable **one-way communication** from backend to frontend. Backend emits events during workflow execution, frontend listens and updates UI reactively.

**Best for:**
- Progress updates (batch processing, long-running tasks)
- Status changes (workflow started, step completed, workflow finished)
- Real-time notifications (new data available, user mentions)
- Log streaming (workflow execution logs)

---

### Server-Sent Events (SSE) Pattern

**Architecture:** HTTP connection stays open, backend streams events as they occur.

**Backend (FastAPI):**

```python
# app/routes/workflows.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from namel3ss import Runtime
import asyncio
import json

router = APIRouter()

@router.post("/workflows/{workflow_id}/execute-stream")
async def execute_workflow_stream(
    workflow_id: str,
    request: ExecuteRequest,
    user: dict = Depends(verify_token)
):
    async def event_generator():
        runtime = Runtime()
        
        # Event callback to send progress updates
        async def on_event(event: dict):
            # Format as SSE
            yield f"event: {event['type']}\n"
            yield f"data: {json.dumps(event['data'])}\n\n"
        
        # Register event handler
        runtime.on('workflow_started', lambda data: on_event({'type': 'workflow_started', 'data': data}))
        runtime.on('step_started', lambda data: on_event({'type': 'step_started', 'data': data}))
        runtime.on('step_completed', lambda data: on_event({'type': 'step_completed', 'data': data}))
        runtime.on('progress_update', lambda data: on_event({'type': 'progress_update', 'data': data}))
        runtime.on('workflow_completed', lambda data: on_event({'type': 'workflow_completed', 'data': data}))
        runtime.on('workflow_failed', lambda data: on_event({'type': 'workflow_failed', 'data': data}))
        
        try:
            # Execute workflow
            result = await runtime.execute_workflow(
                workflow_id,
                input=request.input,
                context={'user_id': user['user_id']}
            )
            
            # Final event
            yield f"event: workflow_completed\n"
            yield f"data: {json.dumps({'execution_id': result.id, 'output': result.output})}\n\n"
            
        except Exception as error:
            yield f"event: workflow_failed\n"
            yield f"data: {json.dumps({'error': str(error)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

**Frontend (React):**

```typescript
// hooks/useWorkflowStream.ts
import { useEffect, useState } from 'react';

export interface WorkflowEvent {
  type: 'workflow_started' | 'step_started' | 'step_completed' | 'progress_update' | 'workflow_completed' | 'workflow_failed';
  data: any;
}

export function useWorkflowStream(workflowId: string, input: any) {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    let eventSource: EventSource | null = null;

    const startStream = async () => {
      setIsStreaming(true);
      setError(null);
      setEvents([]);

      const token = localStorage.getItem('auth_token');
      
      // Start SSE connection
      eventSource = new EventSource(
        `https://api.example.com/workflows/${workflowId}/execute-stream?` +
        `token=${token}&input=${encodeURIComponent(JSON.stringify(input))}`
      );

      // Listen for all event types
      ['workflow_started', 'step_started', 'step_completed', 'progress_update', 'workflow_completed', 'workflow_failed'].forEach((eventType) => {
        eventSource!.addEventListener(eventType, (event) => {
          const data = JSON.parse(event.data);
          
          setEvents((prev) => [...prev, { type: eventType as any, data }]);

          if (eventType === 'workflow_completed') {
            setResult(data);
            setIsStreaming(false);
            eventSource?.close();
          } else if (eventType === 'workflow_failed') {
            setError(new Error(data.error));
            setIsStreaming(false);
            eventSource?.close();
          }
        });
      });

      eventSource.onerror = (err) => {
        console.error('SSE error:', err);
        setError(new Error('Connection lost'));
        setIsStreaming(false);
        eventSource?.close();
      };
    };

    startStream();

    return () => {
      eventSource?.close();
    };
  }, [workflowId, input]);

  return { events, isStreaming, error, result };
}
```

**Usage in Component:**

```typescript
// components/BatchImportProgress.tsx
import { useWorkflowStream } from '@/hooks/useWorkflowStream';

export function BatchImportProgress({ fileIds }: { fileIds: number[] }) {
  const { events, isStreaming, error, result } = useWorkflowStream('batch_import', { file_ids: fileIds });

  // Calculate progress from events
  const progress = events.filter((e) => e.type === 'step_completed').length;
  const total = fileIds.length;
  const percent = total > 0 ? (progress / total) * 100 : 0;

  return (
    <div className="max-w-md mx-auto p-6">
      <h2 className="text-xl font-semibold mb-4">Importing Files</h2>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>{progress} / {total} files</span>
          <span>{percent.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      {/* Event log */}
      <div className="bg-gray-50 rounded p-4 max-h-64 overflow-y-auto">
        {events.map((event, i) => (
          <div key={i} className="text-sm mb-2">
            <span className="text-gray-500">{event.type}:</span>{' '}
            <span className="font-mono">{JSON.stringify(event.data)}</span>
          </div>
        ))}
      </div>

      {/* Status */}
      {isStreaming && (
        <div className="mt-4 flex items-center text-blue-600">
          <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Processing...
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          Error: {error.message}
        </div>
      )}

      {result && (
        <div className="mt-4 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
          ✓ Import completed successfully
        </div>
      )}
    </div>
  );
}
```

---

### WebSocket Event Streaming

**Architecture:** Bidirectional WebSocket connection for events + tool calls.

**Backend (FastAPI):**

```python
# app/websocket.py
from fastapi import WebSocket, WebSocketDisconnect
from namel3ss import Runtime
import json

class WorkflowEventEmitter:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
    
    async def emit(self, event_type: str, data: dict):
        await self.websocket.send_json({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    user = verify_websocket_token(token)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    await websocket.accept()
    
    emitter = WorkflowEventEmitter(websocket)
    runtime = Runtime()
    
    # Register event handlers
    runtime.on('workflow_started', lambda data: emitter.emit('workflow_started', data))
    runtime.on('step_started', lambda data: emitter.emit('step_started', data))
    runtime.on('step_completed', lambda data: emitter.emit('step_completed', data))
    runtime.on('progress_update', lambda data: emitter.emit('progress_update', data))
    runtime.on('workflow_completed', lambda data: emitter.emit('workflow_completed', data))
    runtime.on('workflow_failed', lambda data: emitter.emit('workflow_failed', data))
    
    try:
        while True:
            msg = await websocket.receive_json()
            
            if msg["type"] == "execute_workflow":
                workflow_id = msg["workflow_id"]
                input_data = msg["input"]
                
                try:
                    result = await runtime.execute_workflow(
                        workflow_id,
                        input=input_data,
                        context={"user_id": user["user_id"]}
                    )
                    
                    await emitter.emit("workflow_completed", {
                        "execution_id": result.id,
                        "output": result.output
                    })
                    
                except Exception as error:
                    await emitter.emit("workflow_failed", {
                        "error": str(error),
                        "error_type": type(error).__name__
                    })
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user['user_id']}")
```

**Frontend (React):**

```typescript
// hooks/useWorkflowWebSocket.ts
import { useEffect, useState, useRef } from 'react';

export function useWorkflowWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    const ws = new WebSocket(`wss://api.example.com/ws?token=${token}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      setEvents((prev) => [...prev, msg]);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const executeWorkflow = (workflowId: string, input: any) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    wsRef.current.send(
      JSON.stringify({
        type: 'execute_workflow',
        workflow_id: workflowId,
        input,
      })
    );
  };

  const clearEvents = () => setEvents([]);

  return { isConnected, events, executeWorkflow, clearEvents };
}
```

---

### Custom Event Hooks

**Pattern:** Create specialized hooks for specific workflow event patterns.

#### Hook 1: Progress Tracking

```typescript
// hooks/useWorkflowProgress.ts
import { useWorkflowWebSocket } from './useWorkflowWebSocket';
import { useMemo } from 'react';

export function useWorkflowProgress(executionId?: string) {
  const { events } = useWorkflowWebSocket();

  const progress = useMemo(() => {
    if (!executionId) return null;

    const workflowEvents = events.filter(
      (e) => e.data?.execution_id === executionId
    );

    const progressEvents = workflowEvents.filter(
      (e) => e.type === 'progress_update'
    );

    if (progressEvents.length === 0) return null;

    const latest = progressEvents[progressEvents.length - 1];
    return {
      current: latest.data.current,
      total: latest.data.total,
      percent: (latest.data.current / latest.data.total) * 100,
      message: latest.data.message,
    };
  }, [events, executionId]);

  return progress;
}

// Usage
function MyComponent({ executionId }: { executionId: string }) {
  const progress = useWorkflowProgress(executionId);

  if (!progress) return <div>Initializing...</div>;

  return (
    <div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-600 h-2 rounded-full"
          style={{ width: `${progress.percent}%` }}
        />
      </div>
      <p className="text-sm text-gray-600 mt-2">{progress.message}</p>
    </div>
  );
}
```

#### Hook 2: Workflow Status

```typescript
// hooks/useWorkflowStatus.ts
import { useWorkflowWebSocket } from './useWorkflowWebSocket';
import { useMemo } from 'react';

type WorkflowStatus = 'idle' | 'running' | 'completed' | 'failed';

export function useWorkflowStatus(executionId?: string) {
  const { events } = useWorkflowWebSocket();

  const status = useMemo<{
    status: WorkflowStatus;
    currentStep?: string;
    error?: string;
    result?: any;
  }>(() => {
    if (!executionId) return { status: 'idle' };

    const workflowEvents = events.filter(
      (e) => e.data?.execution_id === executionId
    );

    // Check for completion
    const completed = workflowEvents.find((e) => e.type === 'workflow_completed');
    if (completed) {
      return {
        status: 'completed',
        result: completed.data.output,
      };
    }

    // Check for failure
    const failed = workflowEvents.find((e) => e.type === 'workflow_failed');
    if (failed) {
      return {
        status: 'failed',
        error: failed.data.error,
      };
    }

    // Check for running
    const stepEvents = workflowEvents.filter(
      (e) => e.type === 'step_started' || e.type === 'step_completed'
    );

    if (stepEvents.length > 0) {
      const lastStep = stepEvents[stepEvents.length - 1];
      return {
        status: 'running',
        currentStep: lastStep.data.step_id,
      };
    }

    return { status: 'idle' };
  }, [events, executionId]);

  return status;
}

// Usage
function WorkflowStatusBadge({ executionId }: { executionId: string }) {
  const { status, currentStep, error } = useWorkflowStatus(executionId);

  const statusConfig = {
    idle: { label: 'Idle', color: 'bg-gray-500' },
    running: { label: 'Running', color: 'bg-blue-500' },
    completed: { label: 'Completed', color: 'bg-green-500' },
    failed: { label: 'Failed', color: 'bg-red-500' },
  };

  const config = statusConfig[status];

  return (
    <div>
      <span className={`px-2 py-1 rounded text-white text-sm ${config.color}`}>
        {config.label}
      </span>
      {currentStep && (
        <span className="ml-2 text-sm text-gray-600">Step: {currentStep}</span>
      )}
      {error && (
        <div className="mt-2 text-sm text-red-600">{error}</div>
      )}
    </div>
  );
}
```

#### Hook 3: Event Log Viewer

```typescript
// hooks/useWorkflowLogs.ts
import { useWorkflowWebSocket } from './useWorkflowWebSocket';
import { useMemo } from 'react';

export function useWorkflowLogs(executionId?: string) {
  const { events } = useWorkflowWebSocket();

  const logs = useMemo(() => {
    if (!executionId) return [];

    return events
      .filter((e) => e.data?.execution_id === executionId)
      .map((e) => ({
        timestamp: e.timestamp,
        type: e.type,
        message: formatEventMessage(e),
        data: e.data,
      }));
  }, [events, executionId]);

  return logs;
}

function formatEventMessage(event: WorkflowEvent): string {
  switch (event.type) {
    case 'workflow_started':
      return `Workflow started`;
    case 'step_started':
      return `Step ${event.data.step_id} started`;
    case 'step_completed':
      return `Step ${event.data.step_id} completed in ${event.data.duration}ms`;
    case 'progress_update':
      return `Progress: ${event.data.current}/${event.data.total} - ${event.data.message}`;
    case 'workflow_completed':
      return `Workflow completed successfully`;
    case 'workflow_failed':
      return `Workflow failed: ${event.data.error}`;
    default:
      return event.type;
  }
}

// Usage
function WorkflowLogViewer({ executionId }: { executionId: string }) {
  const logs = useWorkflowLogs(executionId);

  return (
    <div className="bg-gray-900 text-gray-100 p-4 rounded font-mono text-sm max-h-96 overflow-y-auto">
      {logs.map((log, i) => (
        <div key={i} className="mb-1">
          <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
          <span className="text-blue-400">{log.type}</span> {log.message}
        </div>
      ))}
    </div>
  );
}
```

---

### Event Filtering & Performance

**Challenge:** Long-running workflows generate hundreds of events, causing memory/performance issues.

**Solution 1: Event Buffer Limits**

```typescript
// hooks/useWorkflowWebSocket.ts (optimized)
export function useWorkflowWebSocket(maxEvents: number = 1000) {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);

  // ... WebSocket setup

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    setEvents((prev) => {
      const updated = [...prev, msg];
      
      // Keep only last N events
      if (updated.length > maxEvents) {
        return updated.slice(-maxEvents);
      }
      
      return updated;
    });
  };

  // ...
}
```

**Solution 2: Event Type Filtering**

```typescript
// hooks/useWorkflowWebSocket.ts (with filtering)
export function useWorkflowWebSocket(
  eventFilter?: (event: WorkflowEvent) => boolean
) {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    // Apply filter if provided
    if (eventFilter && !eventFilter(msg)) {
      return;
    }
    
    setEvents((prev) => [...prev, msg]);
  };

  // ...
}

// Usage: Only store progress and completion events
const { events } = useWorkflowWebSocket((event) =>
  ['progress_update', 'workflow_completed', 'workflow_failed'].includes(event.type)
);
```

**Solution 3: Execution-Scoped Events**

```typescript
// hooks/useExecutionEvents.ts
export function useExecutionEvents(executionId: string) {
  const { events: allEvents } = useWorkflowWebSocket();
  
  // Only events for this execution
  const executionEvents = useMemo(
    () => allEvents.filter((e) => e.data?.execution_id === executionId),
    [allEvents, executionId]
  );
  
  return executionEvents;
}
```

---

## Section 3: Browser API Escape Hatches

Workflows that need access to browser-only APIs (camera, geolocation, clipboard, etc.) must delegate those operations to the frontend. This section covers how to expose browser APIs as N3 tools.

---

### Tool 4: Camera Access

**Use Case:** Workflow needs user to take photo (ID verification, barcode scanning, visual inspection).

**N3 Workflow:**
```python
# id_verification.ai
workflow IDVerification {
  steps: [
    {
      "id": "capture_photo",
      "tool": "browser.capturePhoto",
      "args": {
        "title": "Take Photo of ID",
        "instructions": "Center your ID in the frame",
        "facingMode": "environment",  # "user" for selfie, "environment" for back camera
        "maxWidth": 1920,
        "maxHeight": 1080
      }
    },
    {
      "id": "upload_photo",
      "tool": "storage.upload",
      "args": {
        "file_data": steps.capture_photo.image_data,
        "file_name": "id_photo.jpg"
      }
    },
    {
      "id": "verify_id",
      "tool": "ocr.verify_id",
      "args": {"image_url": steps.upload_photo.url}
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/camera-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { useState, useRef, useEffect } from 'react';

interface CameraArgs {
  title: string;
  instructions?: string;
  facingMode: 'user' | 'environment';
  maxWidth?: number;
  maxHeight?: number;
}

interface CameraResult {
  image_data: string;  // base64
  width: number;
  height: number;
  mime_type: string;
}

let cameraResolver: ((result: CameraResult) => void) | null = null;
let cameraRejecter: ((error: Error) => void) | null = null;

export const useCameraTool = (): UITool => {
  return {
    id: 'browser.capturePhoto',
    schema: {
      title: { type: 'string' },
      instructions: { type: 'string' },
      facingMode: { type: 'string', enum: ['user', 'environment'] },
      maxWidth: { type: 'number' },
      maxHeight: { type: 'number' },
    },
    handler: async (args: CameraArgs) => {
      // Check camera permission
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' as PermissionName });
        if (permissionStatus.state === 'denied') {
          throw new Error('Camera permission denied');
        }
      } catch (error) {
        console.warn('Permission API not supported');
      }

      return new Promise((resolve, reject) => {
        cameraResolver = resolve;
        cameraRejecter = reject;

        window.dispatchEvent(
          new CustomEvent('n3:capturePhoto', { detail: args })
        );
      });
    },
  };
};

export function N3CameraContainer() {
  const [isOpen, setIsOpen] = useState(false);
  const [cameraData, setCameraData] = useState<CameraArgs | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const handleCaptureRequest = (event: CustomEvent<CameraArgs>) => {
      setCameraData(event.detail);
      setIsOpen(true);
      setError(null);
    };

    window.addEventListener('n3:capturePhoto', handleCaptureRequest as any);

    return () => {
      window.removeEventListener('n3:capturePhoto', handleCaptureRequest as any);
    };
  }, []);

  useEffect(() => {
    if (isOpen && cameraData) {
      startCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isOpen, cameraData]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: cameraData!.facingMode,
          width: { ideal: cameraData!.maxWidth || 1920 },
          height: { ideal: cameraData!.maxHeight || 1080 },
        },
      });

      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      setError('Failed to access camera. Please check permissions.');
      if (cameraRejecter) {
        cameraRejecter(new Error('Camera access denied'));
        cameraRejecter = null;
      }
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx?.drawImage(video, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.9);

    // Resolve promise
    if (cameraResolver) {
      cameraResolver({
        image_data: imageData,
        width: canvas.width,
        height: canvas.height,
        mime_type: 'image/jpeg',
      });
      cameraResolver = null;
    }

    // Close modal
    stopCamera();
    setIsOpen(false);
    setCameraData(null);
  };

  const handleCancel = () => {
    if (cameraRejecter) {
      cameraRejecter(new Error('Camera cancelled'));
      cameraRejecter = null;
    }
    stopCamera();
    setIsOpen(false);
    setCameraData(null);
  };

  if (!isOpen || !cameraData) return null;

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      {/* Header */}
      <div className="bg-gray-900 text-white p-4">
        <h2 className="text-xl font-semibold">{cameraData.title}</h2>
        {cameraData.instructions && (
          <p className="text-sm text-gray-300 mt-1">{cameraData.instructions}</p>
        )}
      </div>

      {/* Video preview */}
      <div className="flex-1 flex items-center justify-center bg-black">
        {error ? (
          <div className="text-white text-center p-4">
            <p className="mb-4">{error}</p>
            <button
              onClick={handleCancel}
              className="bg-red-600 text-white px-6 py-2 rounded"
            >
              Close
            </button>
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="max-w-full max-h-full"
          />
        )}
      </div>

      {/* Controls */}
      {!error && (
        <div className="bg-gray-900 p-6 flex justify-center gap-4">
          <button
            onClick={handleCancel}
            className="bg-gray-700 text-white px-6 py-3 rounded-full"
          >
            Cancel
          </button>
          <button
            onClick={capturePhoto}
            className="bg-blue-600 text-white px-8 py-3 rounded-full font-semibold"
          >
            Capture Photo
          </button>
        </div>
      )}

      {/* Hidden canvas for capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
```

---

### Tool 5: Geolocation

**Use Case:** Workflow needs user's current location (store finder, delivery tracking, location-based services).

**N3 Workflow:**
```python
# store_finder.ai
workflow StoreFinder {
  steps: [
    {
      "id": "get_location",
      "tool": "browser.getLocation",
      "args": {
        "enableHighAccuracy": true,
        "timeout_ms": 10000,
        "maximumAge": 300000  # Cache location for 5 minutes
      }
    },
    {
      "id": "find_stores",
      "tool": "stores.findNearby",
      "args": {
        "latitude": steps.get_location.latitude,
        "longitude": steps.get_location.longitude,
        "radius_km": 10
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/geolocation-tool.tsx
import { UITool } from '@/lib/n3-tools';

interface GeolocationArgs {
  enableHighAccuracy?: boolean;
  timeout_ms?: number;
  maximumAge?: number;
}

interface GeolocationResult {
  latitude: number;
  longitude: number;
  accuracy: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
  heading?: number | null;
  speed?: number | null;
  timestamp: number;
}

export const useGeolocationTool = (): UITool => {
  return {
    id: 'browser.getLocation',
    schema: {
      enableHighAccuracy: { type: 'boolean' },
      timeout_ms: { type: 'number' },
      maximumAge: { type: 'number' },
    },
    handler: async (args: GeolocationArgs): Promise<GeolocationResult> => {
      // Check geolocation permission
      if (!navigator.geolocation) {
        throw new Error('Geolocation not supported by browser');
      }

      // Show loading indicator (optional)
      window.dispatchEvent(
        new CustomEvent('n3:geolocationRequested', { detail: args })
      );

      return new Promise((resolve, reject) => {
        const options: PositionOptions = {
          enableHighAccuracy: args.enableHighAccuracy ?? true,
          timeout: args.timeout_ms ?? 10000,
          maximumAge: args.maximumAge ?? 0,
        };

        navigator.geolocation.getCurrentPosition(
          (position) => {
            // Hide loading indicator
            window.dispatchEvent(new CustomEvent('n3:geolocationReceived'));

            resolve({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy,
              altitude: position.coords.altitude,
              altitudeAccuracy: position.coords.altitudeAccuracy,
              heading: position.coords.heading,
              speed: position.coords.speed,
              timestamp: position.timestamp,
            });
          },
          (error) => {
            window.dispatchEvent(new CustomEvent('n3:geolocationError'));

            let errorMessage = 'Failed to get location';
            switch (error.code) {
              case error.PERMISSION_DENIED:
                errorMessage = 'Location permission denied';
                break;
              case error.POSITION_UNAVAILABLE:
                errorMessage = 'Location unavailable';
                break;
              case error.TIMEOUT:
                errorMessage = 'Location request timeout';
                break;
            }

            reject(new Error(errorMessage));
          },
          options
        );
      });
    },
  };
};

// Optional: Loading indicator component
export function GeolocationIndicator() {
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const handleRequested = () => setIsLoading(true);
    const handleReceived = () => setIsLoading(false);
    const handleError = () => setIsLoading(false);

    window.addEventListener('n3:geolocationRequested', handleRequested);
    window.addEventListener('n3:geolocationReceived', handleReceived);
    window.addEventListener('n3:geolocationError', handleError);

    return () => {
      window.removeEventListener('n3:geolocationRequested', handleRequested);
      window.removeEventListener('n3:geolocationReceived', handleReceived);
      window.removeEventListener('n3:geolocationError', handleError);
    };
  }, []);

  if (!isLoading) return null;

  return (
    <div className="fixed top-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 z-50">
      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
      </svg>
      Getting your location...
    </div>
  );
}
```

---

### Tool 6: Clipboard Access

**Use Case:** Workflow generates code, API key, or text that user needs to copy.

**N3 Workflow:**
```python
# api_key_generation.ai
workflow GenerateAPIKey {
  steps: [
    {
      "id": "generate_key",
      "tool": "keys.generate",
      "args": {"user_id": input.user_id}
    },
    {
      "id": "copy_to_clipboard",
      "tool": "browser.copyToClipboard",
      "args": {
        "text": steps.generate_key.api_key,
        "notify": true,
        "notification_message": "API key copied to clipboard"
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/clipboard-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { toast } from 'sonner';

interface ClipboardArgs {
  text: string;
  notify?: boolean;
  notification_message?: string;
}

export const useClipboardTool = (): UITool => {
  return {
    id: 'browser.copyToClipboard',
    schema: {
      text: { type: 'string' },
      notify: { type: 'boolean' },
      notification_message: { type: 'string' },
    },
    handler: async (args: ClipboardArgs) => {
      try {
        // Modern Clipboard API
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(args.text);
        } else {
          // Fallback for older browsers
          const textArea = document.createElement('textarea');
          textArea.value = args.text;
          textArea.style.position = 'fixed';
          textArea.style.left = '-999999px';
          document.body.appendChild(textArea);
          textArea.focus();
          textArea.select();
          
          try {
            document.execCommand('copy');
          } finally {
            document.body.removeChild(textArea);
          }
        }

        // Show notification if requested
        if (args.notify) {
          toast.success(args.notification_message || 'Copied to clipboard');
        }

        return { copied: true };
      } catch (error) {
        toast.error('Failed to copy to clipboard');
        throw new Error('Clipboard access denied');
      }
    },
  };
};
```

---

### Tool 7: Browser Notifications

**Use Case:** Workflow wants to send desktop notification (background job completed, new message).

**N3 Workflow:**
```python
# background_report.ai
workflow GenerateReport {
  steps: [
    {
      "id": "generate",
      "tool": "reports.generate",
      "args": {"type": "sales", "year": 2025}
    },
    {
      "id": "notify_user",
      "tool": "browser.showNotification",
      "args": {
        "title": "Report Ready",
        "body": "Your sales report for 2025 is ready to download",
        "icon": "/icons/report.png",
        "tag": "report_complete",
        "requireInteraction": false
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/notification-tool.tsx
import { UITool } from '@/lib/n3-tools';

interface BrowserNotificationArgs {
  title: string;
  body: string;
  icon?: string;
  tag?: string;
  requireInteraction?: boolean;
  data?: any;
}

export const useBrowserNotificationTool = (): UITool => {
  return {
    id: 'browser.showNotification',
    schema: {
      title: { type: 'string' },
      body: { type: 'string' },
      icon: { type: 'string' },
      tag: { type: 'string' },
      requireInteraction: { type: 'boolean' },
      data: { type: 'object' },
    },
    handler: async (args: BrowserNotificationArgs) => {
      // Check if notifications are supported
      if (!('Notification' in window)) {
        throw new Error('Browser notifications not supported');
      }

      // Request permission if not granted
      if (Notification.permission === 'default') {
        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
          throw new Error('Notification permission denied');
        }
      }

      if (Notification.permission === 'denied') {
        throw new Error('Notification permission denied');
      }

      // Show notification
      const notification = new Notification(args.title, {
        body: args.body,
        icon: args.icon || '/favicon.ico',
        tag: args.tag,
        requireInteraction: args.requireInteraction ?? false,
        data: args.data,
      });

      // Handle notification click
      notification.onclick = (event) => {
        event.preventDefault();
        window.focus();
        notification.close();

        // Emit custom event for app to handle
        window.dispatchEvent(
          new CustomEvent('n3:notificationClicked', {
            detail: { tag: args.tag, data: args.data },
          })
        );
      };

      return { shown: true };
    },
  };
};

// Example: Handle notification clicks
export function useNotificationClickHandler() {
  useEffect(() => {
    const handleClick = (event: CustomEvent) => {
      const { tag, data } = event.detail;

      // Route based on notification tag
      switch (tag) {
        case 'report_complete':
          window.location.href = '/reports';
          break;
        case 'new_message':
          window.location.href = `/messages/${data.message_id}`;
          break;
        // ... more handlers
      }
    };

    window.addEventListener('n3:notificationClicked', handleClick as any);

    return () => {
      window.removeEventListener('n3:notificationClicked', handleClick as any);
    };
  }, []);
}
```

---

### Tool 8: Page Visibility

**Use Case:** Workflow needs to know if user is currently viewing the page (pause video processing, reduce polling).

**N3 Workflow:**
```python
# video_streaming.ai
workflow VideoStream {
  steps: [
    {
      "id": "check_visibility",
      "tool": "browser.isPageVisible",
      "args": {}
    },
    {
      "id": "stream_video",
      "tool": "video.stream",
      "args": {
        "quality": "high" if steps.check_visibility.visible else "low",
        "video_id": input.video_id
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/page-visibility-tool.tsx
import { UITool } from '@/lib/n3-tools';

export const usePageVisibilityTool = (): UITool => {
  return {
    id: 'browser.isPageVisible',
    schema: {},
    handler: async () => {
      return {
        visible: document.visibilityState === 'visible',
        hidden: document.hidden,
        visibility_state: document.visibilityState,
      };
    },
  };
};

// Advanced: Stream visibility changes
export const usePageVisibilityStreamTool = (): UITool => {
  const visibilityListeners = new Map<string, (visible: boolean) => void>();

  return {
    id: 'browser.watchPageVisibility',
    schema: {
      callback_id: { type: 'string' },
    },
    handler: async (args: { callback_id: string }) => {
      // Register visibility change listener
      const handler = () => {
        const callback = visibilityListeners.get(args.callback_id);
        if (callback) {
          callback(document.visibilityState === 'visible');
        }
      };

      document.addEventListener('visibilitychange', handler);
      visibilityListeners.set(args.callback_id, (visible: boolean) => {
        // Send visibility update to backend
        fetch('/api/visibility-update', {
          method: 'POST',
          body: JSON.stringify({
            callback_id: args.callback_id,
            visible,
          }),
        });
      });

      return { watching: true };
    },
  };
};
```

---

### Security Considerations

#### Permission Requests

**Best Practice:** Request permissions at the right time, explain why they're needed.

```typescript
// Request camera permission with context
async function requestCameraPermission(): Promise<boolean> {
  // Show explanation modal first
  const userConfirmed = await showModal({
    title: 'Camera Permission Required',
    message: 'We need access to your camera to verify your identity. Your photos are encrypted and never shared.',
    actions: ['Allow', 'Deny'],
  });

  if (userConfirmed !== 'Allow') {
    return false;
  }

  // Then request actual permission
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach((track) => track.stop());  // Stop immediately
    return true;
  } catch (error) {
    return false;
  }
}
```

#### Data Privacy

**Best Practice:** Never send sensitive data unnecessarily, encrypt when possible.

```typescript
// ❌ BAD: Send full image to backend
const imageData = canvas.toDataURL('image/jpeg');
await sendToBackend({ image_data: imageData });  // 2-5 MB of base64

// ✅ GOOD: Upload to storage first, send URL
const blob = await (await fetch(canvas.toDataURL('image/jpeg'))).blob();
const formData = new FormData();
formData.append('file', blob, 'photo.jpg');

const uploadResponse = await fetch('/api/upload', {
  method: 'POST',
  body: formData,
});
const { file_url } = await uploadResponse.json();

await sendToBackend({ image_url: file_url });  // Only URL, not full image
```

#### Permission Denial Handling

**Best Practice:** Gracefully handle permission denials, provide alternatives.

```typescript
async function handleCameraRequest() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    // ... use camera
  } catch (error) {
    if (error.name === 'NotAllowedError') {
      // User denied permission
      showModal({
        title: 'Camera Permission Denied',
        message: 'You can still verify your identity by uploading a photo from your device.',
        actions: ['Upload Photo', 'Cancel'],
        onAction: (action) => {
          if (action === 'Upload Photo') {
            triggerFileUpload();
          }
        },
      });
    } else if (error.name === 'NotFoundError') {
      // No camera available
      showModal({
        title: 'No Camera Found',
        message: 'Please upload a photo from your device instead.',
        actions: ['Upload Photo'],
        onAction: () => triggerFileUpload(),
      });
    }
  }
}
```

---

## Section 4: Navigation & State Management Escape Hatches

Workflows often need to control frontend navigation (redirect to success page, open modal, update URL) or manage client-side state. This section covers how workflows can trigger navigation and state changes from the backend.

---

### Tool 9: Client-Side Navigation

**Use Case:** Workflow completes, redirect user to results page or dashboard.

**N3 Workflow:**
```python
# payment_workflow.ai
workflow ProcessPayment {
  steps: [
    {
      "id": "charge_card",
      "tool": "payments.charge",
      "args": {"amount": input.amount, "card_id": input.card_id}
    },
    {
      "id": "navigate_to_success",
      "tool": "browser.navigate",
      "args": {
        "url": "/payment/success",
        "query": {
          "transaction_id": steps.charge_card.transaction_id,
          "amount": input.amount
        },
        "replace": false  # Add to history (back button works)
      }
    }
  ],
  "on_error": {
    "tool": "browser.navigate",
    "args": {
      "url": "/payment/failed",
      "query": {"error": "error.message"},
      "replace": true  # Replace history (prevent back to payment form)
    }
  }
}
```

**Frontend Implementation:**

```typescript
// components/ui/navigation-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { useRouter } from 'next/navigation';

interface NavigateArgs {
  url: string;
  query?: Record<string, string | number>;
  replace?: boolean;
  openInNewTab?: boolean;
}

export const useNavigationTool = (): UITool => {
  const router = useRouter();

  return {
    id: 'browser.navigate',
    schema: {
      url: { type: 'string' },
      query: { type: 'object' },
      replace: { type: 'boolean' },
      openInNewTab: { type: 'boolean' },
    },
    handler: async (args: NavigateArgs) => {
      // Build URL with query parameters
      let targetUrl = args.url;
      
      if (args.query && Object.keys(args.query).length > 0) {
        const params = new URLSearchParams();
        Object.entries(args.query).forEach(([key, value]) => {
          params.append(key, String(value));
        });
        targetUrl = `${args.url}?${params.toString()}`;
      }

      // Open in new tab if requested
      if (args.openInNewTab) {
        window.open(targetUrl, '_blank', 'noopener,noreferrer');
        return { navigated: true, opened_new_tab: true };
      }

      // Navigate using Next.js router
      if (args.replace) {
        router.replace(targetUrl);
      } else {
        router.push(targetUrl);
      }

      return { navigated: true, url: targetUrl };
    },
  };
};

// React Router version
export const useNavigationToolReactRouter = (): UITool => {
  const navigate = useNavigate();

  return {
    id: 'browser.navigate',
    schema: {
      url: { type: 'string' },
      query: { type: 'object' },
      replace: { type: 'boolean' },
    },
    handler: async (args: NavigateArgs) => {
      let targetUrl = args.url;
      
      if (args.query) {
        const params = new URLSearchParams();
        Object.entries(args.query).forEach(([key, value]) => {
          params.append(key, String(value));
        });
        targetUrl = `${args.url}?${params.toString()}`;
      }

      navigate(targetUrl, { replace: args.replace });

      return { navigated: true, url: targetUrl };
    },
  };
};
```

---

### Tool 10: URL State Management

**Use Case:** Workflow updates URL query parameters without full navigation (filters, pagination, search state).

**N3 Workflow:**
```python
# search_workflow.ai
workflow SearchProducts {
  steps: [
    {
      "id": "search",
      "tool": "products.search",
      "args": {"query": input.query, "filters": input.filters}
    },
    {
      "id": "update_url",
      "tool": "browser.updateURL",
      "args": {
        "query": {
          "q": input.query,
          "category": input.filters.category,
          "page": 1
        },
        "replace": true  # Don't add to history for URL updates
      }
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/url-state-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';

interface UpdateURLArgs {
  query?: Record<string, string | number | null>;  // null removes param
  hash?: string;
  replace?: boolean;
}

export const useURLStateTool = (): UITool => {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  return {
    id: 'browser.updateURL',
    schema: {
      query: { type: 'object' },
      hash: { type: 'string' },
      replace: { type: 'boolean' },
    },
    handler: async (args: UpdateURLArgs) => {
      // Start with current search params
      const newParams = new URLSearchParams(searchParams.toString());

      // Update query parameters
      if (args.query) {
        Object.entries(args.query).forEach(([key, value]) => {
          if (value === null) {
            newParams.delete(key);  // Remove parameter
          } else {
            newParams.set(key, String(value));
          }
        });
      }

      // Build new URL
      let newUrl = pathname;
      const queryString = newParams.toString();
      if (queryString) {
        newUrl += `?${queryString}`;
      }
      if (args.hash) {
        newUrl += `#${args.hash}`;
      }

      // Update URL
      if (args.replace) {
        router.replace(newUrl, { scroll: false });
      } else {
        router.push(newUrl, { scroll: false });
      }

      return { updated: true, url: newUrl };
    },
  };
};
```

---

### Tool 11: Frontend State Updates

**Use Case:** Workflow needs to update React state, Redux store, or global context.

**N3 Workflow:**
```python
# cart_workflow.ai
workflow AddToCart {
  steps: [
    {
      "id": "add_item",
      "tool": "cart.add",
      "args": {"product_id": input.product_id, "quantity": input.quantity}
    },
    {
      "id": "update_cart_state",
      "tool": "browser.updateState",
      "args": {
        "store": "cart",
        "action": "SET_ITEMS",
        "payload": {"items": steps.add_item.cart_items, "total": steps.add_item.total}
      }
    },
    {
      "id": "show_notification",
      "tool": "ui.notify",
      "args": {
        "message": "Item added to cart",
        "type": "success"
      }
    }
  ]
}
```

**Frontend Implementation (Redux):**

```typescript
// components/ui/state-update-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { useDispatch } from 'react-redux';

interface StateUpdateArgs {
  store: string;
  action: string;
  payload: any;
}

export const useStateUpdateTool = (): UITool => {
  const dispatch = useDispatch();

  return {
    id: 'browser.updateState',
    schema: {
      store: { type: 'string' },
      action: { type: 'string' },
      payload: { type: 'object' },
    },
    handler: async (args: StateUpdateArgs) => {
      // Dispatch Redux action
      dispatch({
        type: `${args.store.toUpperCase()}_${args.action}`,
        payload: args.payload,
      });

      return { updated: true, store: args.store, action: args.action };
    },
  };
};
```

**Frontend Implementation (Zustand):**

```typescript
// lib/store.ts
import { create } from 'zustand';

interface AppState {
  cart: {
    items: any[];
    total: number;
  };
  user: {
    id: string;
    name: string;
  } | null;
  setCart: (cart: { items: any[]; total: number }) => void;
  setUser: (user: { id: string; name: string } | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  cart: { items: [], total: 0 },
  user: null,
  setCart: (cart) => set({ cart }),
  setUser: (user) => set({ user }),
}));

// components/ui/state-update-tool.tsx (Zustand version)
import { UITool } from '@/lib/n3-tools';
import { useAppStore } from '@/lib/store';

export const useStateUpdateToolZustand = (): UITool => {
  const store = useAppStore();

  return {
    id: 'browser.updateState',
    schema: {
      store: { type: 'string' },
      action: { type: 'string' },
      payload: { type: 'object' },
    },
    handler: async (args: StateUpdateArgs) => {
      // Route to appropriate store setter
      switch (args.store) {
        case 'cart':
          if (args.action === 'SET_ITEMS') {
            store.setCart(args.payload);
          }
          break;
        case 'user':
          if (args.action === 'SET_USER') {
            store.setUser(args.payload);
          }
          break;
        default:
          throw new Error(`Unknown store: ${args.store}`);
      }

      return { updated: true, store: args.store };
    },
  };
};
```

**Frontend Implementation (React Context):**

```typescript
// contexts/AppContext.tsx
import { createContext, useContext, useState, ReactNode } from 'react';

interface AppContextValue {
  cart: { items: any[]; total: number };
  setCart: (cart: { items: any[]; total: number }) => void;
  user: { id: string; name: string } | null;
  setUser: (user: { id: string; name: string } | null) => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [cart, setCart] = useState({ items: [], total: 0 });
  const [user, setUser] = useState(null);

  return (
    <AppContext.Provider value={{ cart, setCart, user, setUser }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
}

// components/ui/state-update-tool.tsx (Context version)
import { UITool } from '@/lib/n3-tools';
import { useAppContext } from '@/contexts/AppContext';

export const useStateUpdateToolContext = (): UITool => {
  const { setCart, setUser } = useAppContext();

  return {
    id: 'browser.updateState',
    schema: {
      store: { type: 'string' },
      action: { type: 'string' },
      payload: { type: 'object' },
    },
    handler: async (args: StateUpdateArgs) => {
      switch (args.store) {
        case 'cart':
          setCart(args.payload);
          break;
        case 'user':
          setUser(args.payload);
          break;
        default:
          throw new Error(`Unknown store: ${args.store}`);
      }

      return { updated: true };
    },
  };
};
```

---

### Tool 12: Modal/Dialog Management

**Use Case:** Workflow opens a specific modal or dialog (login modal, upgrade prompt, terms acceptance).

**N3 Workflow:**
```python
# feature_access_workflow.ai
workflow AccessPremiumFeature {
  steps: [
    {
      "id": "check_subscription",
      "tool": "subscriptions.check",
      "args": {"user_id": context.user_id}
    },
    {
      "id": "show_upgrade_modal",
      "tool": "browser.openModal",
      "args": {
        "modal_id": "upgrade_prompt",
        "props": {
          "current_plan": steps.check_subscription.plan,
          "required_plan": "premium",
          "feature": "advanced_analytics"
        }
      },
      "condition": "steps.check_subscription.plan != 'premium'"
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/modal-manager-tool.tsx
import { UITool } from '@/lib/n3-tools';
import { create } from 'zustand';

interface ModalState {
  activeModal: string | null;
  modalProps: Record<string, any>;
  openModal: (modalId: string, props?: Record<string, any>) => void;
  closeModal: () => void;
}

export const useModalStore = create<ModalState>((set) => ({
  activeModal: null,
  modalProps: {},
  openModal: (modalId, props = {}) => set({ activeModal: modalId, modalProps: props }),
  closeModal: () => set({ activeModal: null, modalProps: {} }),
}));

interface OpenModalArgs {
  modal_id: string;
  props?: Record<string, any>;
}

export const useModalManagerTool = (): UITool => {
  const { openModal } = useModalStore();

  return {
    id: 'browser.openModal',
    schema: {
      modal_id: { type: 'string' },
      props: { type: 'object' },
    },
    handler: async (args: OpenModalArgs) => {
      openModal(args.modal_id, args.props);
      return { opened: true, modal_id: args.modal_id };
    },
  };
};

// Modal registry component
export function ModalRegistry() {
  const { activeModal, modalProps, closeModal } = useModalStore();

  // Map modal IDs to components
  const modals: Record<string, React.ComponentType<any>> = {
    upgrade_prompt: UpgradeModal,
    login_required: LoginModal,
    terms_acceptance: TermsModal,
    confirmation: ConfirmationModal,
  };

  const ModalComponent = activeModal ? modals[activeModal] : null;

  if (!ModalComponent) return null;

  return <ModalComponent {...modalProps} onClose={closeModal} />;
}

// Example modal component
function UpgradeModal({
  current_plan,
  required_plan,
  feature,
  onClose,
}: {
  current_plan: string;
  required_plan: string;
  feature: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md">
        <h2 className="text-xl font-semibold mb-2">Upgrade Required</h2>
        <p className="text-gray-700 mb-4">
          The feature "{feature}" requires a {required_plan} plan. You are currently on the {current_plan} plan.
        </p>
        <div className="flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded">
            Cancel
          </button>
          <button
            onClick={() => {
              window.location.href = '/upgrade';
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Upgrade Now
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

### Tool 13: Scroll & Focus Control

**Use Case:** Workflow wants to scroll to specific element or focus on input field (validation errors, tour guides).

**N3 Workflow:**
```python
# form_validation_workflow.ai
workflow ValidateForm {
  steps: [
    {
      "id": "validate",
      "tool": "forms.validate",
      "args": {"form_data": input.form_data}
    },
    {
      "id": "scroll_to_error",
      "tool": "browser.scrollToElement",
      "args": {
        "selector": "[data-field='{{steps.validate.first_error_field}}']",
        "behavior": "smooth",
        "block": "center",
        "focus": true
      },
      "condition": "steps.validate.has_errors"
    }
  ]
}
```

**Frontend Implementation:**

```typescript
// components/ui/scroll-tool.tsx
import { UITool } from '@/lib/n3-tools';

interface ScrollArgs {
  selector?: string;
  elementId?: string;
  behavior?: 'auto' | 'smooth';
  block?: 'start' | 'center' | 'end' | 'nearest';
  inline?: 'start' | 'center' | 'end' | 'nearest';
  focus?: boolean;
}

export const useScrollTool = (): UITool => {
  return {
    id: 'browser.scrollToElement',
    schema: {
      selector: { type: 'string' },
      elementId: { type: 'string' },
      behavior: { type: 'string', enum: ['auto', 'smooth'] },
      block: { type: 'string', enum: ['start', 'center', 'end', 'nearest'] },
      inline: { type: 'string', enum: ['start', 'center', 'end', 'nearest'] },
      focus: { type: 'boolean' },
    },
    handler: async (args: ScrollArgs) => {
      let element: HTMLElement | null = null;

      // Find element by selector or ID
      if (args.selector) {
        element = document.querySelector(args.selector);
      } else if (args.elementId) {
        element = document.getElementById(args.elementId);
      }

      if (!element) {
        throw new Error('Element not found');
      }

      // Scroll to element
      element.scrollIntoView({
        behavior: args.behavior || 'smooth',
        block: args.block || 'center',
        inline: args.inline || 'nearest',
      });

      // Focus element if requested
      if (args.focus && element instanceof HTMLElement) {
        element.focus({ preventScroll: true });
      }

      return { scrolled: true, focused: args.focus };
    },
  };
};
```

---

## Section 5: Production Patterns & Best Practices

This section covers production-grade patterns for building reliable, performant escape hatches at scale.

---

### Error Recovery & Retry Logic

#### Pattern 1: Tool Call Timeout with Retry

**Challenge:** Tool calls may timeout due to network issues, user inactivity, or frontend crashes.

**Solution:**

```typescript
// lib/n3-tools-resilient.ts
import { UITool } from '@/lib/n3-tools';

interface RetryConfig {
  maxRetries: number;
  timeoutMs: number;
  backoffMultiplier: number;
}

export class ResilientToolRegistry extends N3ToolRegistry {
  private retryConfig: RetryConfig = {
    maxRetries: 3,
    timeoutMs: 30000,
    backoffMultiplier: 2,
  };

  async executeToolWithRetry(
    toolId: string,
    args: any,
    attempt: number = 0
  ): Promise<any> {
    const tool = this.tools.get(toolId);
    if (!tool) {
      throw new Error(`Tool not found: ${toolId}`);
    }

    const timeout = this.retryConfig.timeoutMs * Math.pow(
      this.retryConfig.backoffMultiplier,
      attempt
    );

    try {
      // Execute with timeout
      const result = await Promise.race([
        tool.handler(args),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Tool timeout')), timeout)
        ),
      ]);

      return result;
    } catch (error) {
      // Retry logic
      if (attempt < this.retryConfig.maxRetries) {
        console.warn(`Tool ${toolId} failed, retrying (${attempt + 1}/${this.retryConfig.maxRetries})...`);
        
        // Exponential backoff
        await new Promise((resolve) =>
          setTimeout(resolve, 1000 * Math.pow(2, attempt))
        );
        
        return this.executeToolWithRetry(toolId, args, attempt + 1);
      }

      throw error;
    }
  }

  private async handleToolCall(msg: {
    call_id: string;
    tool: string;
    args: any;
    execution_id: string;
  }) {
    const { call_id, tool, args, execution_id } = msg;

    try {
      const result = await this.executeToolWithRetry(tool, args);

      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result,
          error: null,
        })
      );
    } catch (error) {
      console.error(`Tool call failed after retries: ${tool}`, error);

      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result: null,
          error: {
            message: error.message,
            retries_exhausted: true,
          },
        })
      );
    }
  }
}
```

#### Pattern 2: Graceful Degradation

**Challenge:** If a tool fails, workflow should continue with fallback behavior.

**Solution:**

```typescript
// Example: Camera tool with file upload fallback
export const useCameraToolWithFallback = (): UITool => {
  return {
    id: 'browser.capturePhoto',
    schema: {
      title: { type: 'string' },
      fallback_to_upload: { type: 'boolean' },
    },
    handler: async (args: { title: string; fallback_to_upload?: boolean }) => {
      try {
        // Attempt camera access
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        
        // Show camera UI and capture photo
        const photo = await showCameraUI(stream, args.title);
        
        return { image_data: photo, method: 'camera' };
        
      } catch (error) {
        if (args.fallback_to_upload) {
          // Fallback to file upload
          console.warn('Camera failed, falling back to file upload');
          
          const file = await showFileUploadUI({
            title: args.title,
            accept: 'image/*',
          });
          
          return { image_data: file.data, method: 'upload' };
        }
        
        throw error;
      }
    },
  };
};
```

#### Pattern 3: Frontend Reconnection

**Challenge:** WebSocket disconnects during workflow execution.

**Solution:**

```typescript
// lib/n3-websocket-resilient.ts
export class ResilientWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectInterval = 1000;
  private messageQueue: any[] = [];
  private isReconnecting = false;

  constructor(private url: string, private token: string) {
    this.connect();
  }

  private connect() {
    this.ws = new WebSocket(`${this.url}?token=${this.token}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.isReconnecting = false;

      // Flush message queue
      while (this.messageQueue.length > 0) {
        const msg = this.messageQueue.shift();
        this.send(msg);
      }
    };

    this.ws.onclose = () => {
      console.warn('WebSocket disconnected');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private attemptReconnect() {
    if (this.isReconnecting) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_attempts');
      return;
    }

    this.isReconnecting = true;
    this.reconnectAttempts++;

    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})...`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // Queue message for later
      console.warn('WebSocket not ready, queueing message');
      this.messageQueue.push(data);
    }
  }

  private emit(event: string) {
    window.dispatchEvent(new CustomEvent(`websocket:${event}`));
  }
}

// Usage: Show reconnecting UI
export function WebSocketStatusIndicator() {
  const [status, setStatus] = useState<'connected' | 'reconnecting' | 'failed'>('connected');

  useEffect(() => {
    const handleDisconnect = () => setStatus('reconnecting');
    const handleMaxAttempts = () => setStatus('failed');
    const handleConnect = () => setStatus('connected');

    window.addEventListener('websocket:close', handleDisconnect);
    window.addEventListener('websocket:max_reconnect_attempts', handleMaxAttempts);
    window.addEventListener('websocket:open', handleConnect);

    return () => {
      window.removeEventListener('websocket:close', handleDisconnect);
      window.removeEventListener('websocket:max_reconnect_attempts', handleMaxAttempts);
      window.removeEventListener('websocket:open', handleConnect);
    };
  }, []);

  if (status === 'connected') return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-50">
      {status === 'reconnecting' && (
        <div className="bg-yellow-500 text-white px-4 py-2 text-center">
          Connection lost. Reconnecting...
        </div>
      )}
      {status === 'failed' && (
        <div className="bg-red-600 text-white px-4 py-2 text-center">
          Connection failed. Please refresh the page.
        </div>
      )}
    </div>
  );
}
```

---

### Testing Strategies

#### Unit Testing Tool Handlers

```typescript
// __tests__/tools/camera-tool.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useCameraTool } from '@/components/ui/camera-tool';

// Mock browser APIs
const mockGetUserMedia = vi.fn();
Object.defineProperty(global.navigator, 'mediaDevices', {
  value: {
    getUserMedia: mockGetUserMedia,
  },
  writable: true,
});

describe('Camera Tool', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should request camera with correct constraints', async () => {
    const tool = useCameraTool();
    
    const mockStream = { getTracks: vi.fn(() => []) };
    mockGetUserMedia.mockResolvedValue(mockStream);

    // Trigger tool handler
    window.dispatchEvent(
      new CustomEvent('n3:capturePhoto', {
        detail: {
          title: 'Test',
          facingMode: 'environment',
          maxWidth: 1920,
          maxHeight: 1080,
        },
      })
    );

    // Wait for async operations
    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      video: {
        facingMode: 'environment',
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
  });

  it('should handle permission denial', async () => {
    const tool = useCameraTool();
    
    mockGetUserMedia.mockRejectedValue(new Error('Permission denied'));

    const handlerPromise = tool.handler({
      title: 'Test',
      facingMode: 'user',
    });

    await expect(handlerPromise).rejects.toThrow('Camera access denied');
  });
});
```

#### Integration Testing with WebSocket

```typescript
// __tests__/integration/workflow-tools.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebSocket, Server as WebSocketServer } from 'ws';

describe('Workflow Tool Integration', () => {
  let wss: WebSocketServer;
  let wsClient: WebSocket;

  beforeAll((done) => {
    // Start mock WebSocket server
    wss = new WebSocketServer({ port: 8080 });

    wss.on('connection', (ws) => {
      ws.on('message', (data) => {
        const msg = JSON.parse(data.toString());

        // Simulate tool call request
        if (msg.type === 'register_tools') {
          ws.send(
            JSON.stringify({
              type: 'tool_call_requested',
              call_id: 'test-123',
              tool: 'ui.showModal',
              args: {
                title: 'Test Modal',
                message: 'Integration test',
                actions: ['OK'],
              },
              execution_id: 'exec-456',
            })
          );
        }
      });
    });

    done();
  });

  afterAll(() => {
    wss.close();
  });

  it('should handle tool call and respond', async () => {
    wsClient = new WebSocket('ws://localhost:8080');

    await new Promise((resolve) => {
      wsClient.on('open', resolve);
    });

    // Register tools
    wsClient.send(
      JSON.stringify({
        type: 'register_tools',
        tools: [{ id: 'ui.showModal', schema: {} }],
      })
    );

    // Wait for tool call request
    const response = await new Promise((resolve) => {
      wsClient.on('message', (data) => {
        const msg = JSON.parse(data.toString());
        if (msg.type === 'tool_call_requested') {
          // Simulate user clicking "OK"
          wsClient.send(
            JSON.stringify({
              type: 'tool_call_response',
              call_id: msg.call_id,
              execution_id: msg.execution_id,
              result: 'OK',
              error: null,
            })
          );
        }
        resolve(msg);
      });
    });

    expect(response).toMatchObject({
      type: 'tool_call_requested',
      tool: 'ui.showModal',
    });
  });
});
```

#### E2E Testing with Playwright

```typescript
// e2e/workflow-escape-hatches.spec.ts
import { test, expect } from '@playwright/test';

test('approval workflow with modal escape hatch', async ({ page }) => {
  // Navigate to workflow trigger page
  await page.goto('http://localhost:3000/workflows/approval-demo');

  // Start workflow
  await page.click('button:has-text("Start Approval Workflow")');

  // Wait for modal to appear (triggered by ui.showModal tool)
  const modal = page.locator('[data-testid="n3-modal"]');
  await expect(modal).toBeVisible({ timeout: 5000 });

  // Verify modal content
  await expect(modal.locator('h2')).toHaveText('Approve Payment');
  await expect(modal.locator('.amount')).toHaveText('$5,000');

  // Click approve button
  await modal.locator('button:has-text("Approve")').click();

  // Modal should close
  await expect(modal).not.toBeVisible();

  // Workflow should complete
  await expect(page.locator('.workflow-status')).toHaveText('Completed', { timeout: 10000 });
});

test('camera tool with fallback', async ({ page, context }) => {
  // Deny camera permission
  await context.grantPermissions([]);

  await page.goto('http://localhost:3000/workflows/id-verification');
  await page.click('button:has-text("Verify ID")');

  // Camera should fail, fallback to file upload
  const fileInput = page.locator('input[type="file"]');
  await expect(fileInput).toBeVisible({ timeout: 5000 });

  // Upload file
  await fileInput.setInputFiles('./fixtures/test-id.jpg');

  // Workflow should continue
  await expect(page.locator('.workflow-status')).toHaveText('Completed');
});
```

---

### Performance Optimization

#### Pattern 1: Debounced Tool Calls

**Challenge:** Rapid tool calls (e.g., search as you type) overwhelm backend.

**Solution:**

```typescript
// lib/debounced-tools.ts
import { UITool } from '@/lib/n3-tools';

export function createDebouncedTool(
  tool: UITool,
  debounceMs: number = 300
): UITool {
  let timeoutId: NodeJS.Timeout | null = null;
  let pendingResolve: ((result: any) => void) | null = null;

  return {
    ...tool,
    handler: async (args: any) => {
      // Cancel previous pending call
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      return new Promise((resolve, reject) => {
        pendingResolve = resolve;

        timeoutId = setTimeout(async () => {
          try {
            const result = await tool.handler(args);
            resolve(result);
          } catch (error) {
            reject(error);
          }
        }, debounceMs);
      });
    },
  };
}

// Usage
const searchTool = createDebouncedTool(
  {
    id: 'ui.updateSearch',
    schema: { query: { type: 'string' } },
    handler: async (args) => {
      // Update search UI
      updateSearchResults(args.query);
      return { updated: true };
    },
  },
  300 // 300ms debounce
);
```

#### Pattern 2: Tool Call Batching

**Challenge:** Multiple tool calls in quick succession create network overhead.

**Solution:**

```typescript
// lib/batched-tools.ts
export class BatchedToolRegistry extends N3ToolRegistry {
  private batchQueue: Array<{ tool: string; args: any; resolve: Function; reject: Function }> = [];
  private batchTimeoutId: NodeJS.Timeout | null = null;
  private batchWindowMs = 50;

  async executeTool(tool: string, args: any): Promise<any> {
    return new Promise((resolve, reject) => {
      // Add to batch queue
      this.batchQueue.push({ tool, args, resolve, reject });

      // Schedule batch execution
      if (!this.batchTimeoutId) {
        this.batchTimeoutId = setTimeout(() => {
          this.flushBatch();
        }, this.batchWindowMs);
      }
    });
  }

  private async flushBatch() {
    const batch = [...this.batchQueue];
    this.batchQueue = [];
    this.batchTimeoutId = null;

    // Send batched request to backend
    try {
      const response = await fetch('/api/tools/batch', {
        method: 'POST',
        body: JSON.stringify({
          calls: batch.map((item, i) => ({
            id: i,
            tool: item.tool,
            args: item.args,
          })),
        }),
      });

      const results = await response.json();

      // Resolve individual promises
      results.forEach((result: any, i: number) => {
        if (result.error) {
          batch[i].reject(new Error(result.error));
        } else {
          batch[i].resolve(result.data);
        }
      });
    } catch (error) {
      // Reject all
      batch.forEach((item) => item.reject(error));
    }
  }
}
```

#### Pattern 3: Lazy Tool Registration

**Challenge:** Registering all tools upfront slows initial load.

**Solution:**

```typescript
// lib/lazy-tool-registry.ts
export class LazyToolRegistry extends N3ToolRegistry {
  private toolLoaders: Map<string, () => Promise<UITool>> = new Map();
  private loadedTools: Set<string> = new Set();

  registerLazyTool(toolId: string, loader: () => Promise<UITool>) {
    this.toolLoaders.set(toolId, loader);
  }

  private async ensureToolLoaded(toolId: string) {
    if (this.loadedTools.has(toolId)) return;

    const loader = this.toolLoaders.get(toolId);
    if (!loader) {
      throw new Error(`Tool loader not found: ${toolId}`);
    }

    const tool = await loader();
    this.tools.set(toolId, tool);
    this.loadedTools.add(toolId);
  }

  private async handleToolCall(msg: any) {
    const { tool } = msg;

    // Load tool on-demand
    await this.ensureToolLoaded(tool);

    // Execute as normal
    await super.handleToolCall(msg);
  }
}

// Usage
const registry = new LazyToolRegistry(wsUrl, token);

// Register tools lazily
registry.registerLazyTool('ui.camera', async () => {
  const { useCameraTool } = await import('@/components/ui/camera-tool');
  return useCameraTool();
});

registry.registerLazyTool('ui.geolocation', async () => {
  const { useGeolocationTool } = await import('@/components/ui/geolocation-tool');
  return useGeolocationTool();
});
```

---

### Monitoring & Observability

#### Tool Call Metrics

```typescript
// lib/instrumented-tool-registry.ts
import { UITool } from '@/lib/n3-tools';

export class InstrumentedToolRegistry extends N3ToolRegistry {
  private metrics = {
    toolCalls: new Map<string, number>(),
    toolDurations: new Map<string, number[]>(),
    toolErrors: new Map<string, number>(),
  };

  private async handleToolCall(msg: any) {
    const { call_id, tool, args, execution_id } = msg;
    const startTime = performance.now();

    // Increment call counter
    this.metrics.toolCalls.set(tool, (this.metrics.toolCalls.get(tool) || 0) + 1);

    try {
      const toolHandler = this.tools.get(tool);
      if (!toolHandler) {
        throw new Error(`Tool not registered: ${tool}`);
      }

      const result = await toolHandler.handler(args);

      // Record duration
      const duration = performance.now() - startTime;
      const durations = this.metrics.toolDurations.get(tool) || [];
      durations.push(duration);
      this.metrics.toolDurations.set(tool, durations);

      // Send success response
      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result,
          error: null,
        })
      );

      // Log metrics
      this.logMetrics(tool, duration, 'success');
    } catch (error) {
      // Increment error counter
      this.metrics.toolErrors.set(tool, (this.metrics.toolErrors.get(tool) || 0) + 1);

      // Send error response
      this.ws.send(
        JSON.stringify({
          type: 'tool_call_response',
          call_id,
          execution_id,
          result: null,
          error: error.message,
        })
      );

      // Log error
      this.logMetrics(tool, performance.now() - startTime, 'error');
    }
  }

  private logMetrics(tool: string, duration: number, status: 'success' | 'error') {
    // Send to analytics
    if (typeof window !== 'undefined' && (window as any).analytics) {
      (window as any).analytics.track('Tool Call', {
        tool,
        duration,
        status,
        timestamp: Date.now(),
      });
    }

    // Log to console in dev
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Tool Metrics] ${tool}: ${duration.toFixed(2)}ms (${status})`);
    }
  }

  getMetrics() {
    return {
      calls: Object.fromEntries(this.metrics.toolCalls),
      avgDurations: Object.fromEntries(
        Array.from(this.metrics.toolDurations.entries()).map(([tool, durations]) => [
          tool,
          durations.reduce((a, b) => a + b, 0) / durations.length,
        ])
      ),
      errors: Object.fromEntries(this.metrics.toolErrors),
    };
  }
}
```

---

### Summary: Production Checklist

**✅ Before deploying escape hatches to production:**

1. **Error Handling**
   - [ ] Tool timeout handling with configurable limits
   - [ ] Retry logic with exponential backoff
   - [ ] Graceful degradation for critical tools
   - [ ] User-friendly error messages

2. **Performance**
   - [ ] Debouncing for high-frequency tools
   - [ ] Lazy loading for large tool sets
   - [ ] Event buffer limits to prevent memory leaks
   - [ ] WebSocket connection pooling

3. **Security**
   - [ ] Permission requests with user consent
   - [ ] Input validation on tool arguments
   - [ ] Rate limiting on tool calls
   - [ ] Audit logging for sensitive operations

4. **Monitoring**
   - [ ] Tool call success/failure metrics
   - [ ] Duration tracking for performance issues
   - [ ] Error rate alerts
   - [ ] User interaction analytics

5. **Testing**
   - [ ] Unit tests for tool handlers
   - [ ] Integration tests for WebSocket flow
   - [ ] E2E tests for critical workflows
   - [ ] Load testing for concurrent tool calls

6. **Documentation**
   - [ ] Tool schema documentation
   - [ ] Example workflows for each tool
   - [ ] Troubleshooting guide
   - [ ] Migration guide for new tools

---

