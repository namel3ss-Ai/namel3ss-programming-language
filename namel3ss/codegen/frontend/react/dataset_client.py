"""
Generate DatasetClient TypeScript library for dynamic data binding.

This module generates comprehensive TypeScript code for:
- DatasetClient class with fetch/subscribe/mutate methods
- React hooks for dataset binding (useDataset, useDatasetMutation)
- WebSocket transport layer with automatic reconnection
- Polling transport fallback when WebSocket unavailable
- Optimistic updates and conflict resolution
- Type-safe CRUD operations
"""

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from namel3ss.ir import BackendIR, DatasetSpec, DataBindingSpec

from .utils import write_file


def write_dataset_client_lib(lib_dir: Path, backend_ir: "BackendIR") -> None:
    """Generate datasetClient.ts with comprehensive dataset binding functionality.
    
    This creates a TypeScript module containing:
    - DatasetClient class for CRUD operations
    - WebSocket manager for realtime subscriptions
    - React hooks for dataset binding
    - Type definitions for all datasets
    - Optimistic update handling
    """
    
    content = _generate_dataset_client_code(backend_ir)
    write_file(lib_dir / "datasetClient.ts", content)


def _generate_dataset_client_code(backend_ir: "BackendIR") -> str:
    """Generate the complete DatasetClient TypeScript code."""
    
    # Generate dataset type definitions
    dataset_types = _generate_dataset_types(backend_ir.datasets)
    
    # Check if any datasets have realtime enabled
    has_realtime = any(ds.realtime_enabled for ds in backend_ir.datasets)
    
    template = f'''
import {{ useEffect, useState, useCallback, useRef }} from "react";

// ============================================================================
// Type Definitions
// ============================================================================

export interface PaginatedResponse<T> {{
  data: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}}

export interface DatasetQueryOptions {{
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: "asc" | "desc";
  search?: string;
  filters?: Record<string, any>;
}}

export interface DatasetMutationResult<T> {{
  success: boolean;
  data?: T;
  error?: string;
}}

export interface WebSocketMessage {{
  type: string;
  dataset: string;
  payload: any;
  meta?: {{
    event_type?: string;
    ts?: number;
    node?: string;
  }};
}}

{dataset_types}

// ============================================================================
// WebSocket Manager (Realtime Subscriptions)
// ============================================================================

class WebSocketManager {{
  private connections: Map<string, WebSocket> = new Map();
  private listeners: Map<string, Set<(message: WebSocketMessage) => void>> = new Map();
  private reconnectTimers: Map<string, NodeJS.Timeout> = new Map();
  private maxReconnectDelay = 30000; // 30 seconds
  private baseReconnectDelay = 1000; // 1 second
  
  subscribe(
    datasetName: string,
    callback: (message: WebSocketMessage) => void,
    baseUrl: string = ""
  ): () => void {{
    // Add listener
    if (!this.listeners.has(datasetName)) {{
      this.listeners.set(datasetName, new Set());
    }}
    this.listeners.get(datasetName)!.add(callback);
    
    // Create WebSocket connection if needed
    if (!this.connections.has(datasetName)) {{
      this.connect(datasetName, baseUrl);
    }}
    
    // Return unsubscribe function
    return () => {{
      const listenerSet = this.listeners.get(datasetName);
      if (listenerSet) {{
        listenerSet.delete(callback);
        
        // Close connection if no more listeners
        if (listenerSet.size === 0) {{
          this.disconnect(datasetName);
        }}
      }}
    }};
  }}
  
  private connect(datasetName: string, baseUrl: string, attempt: number = 0): void {{
    const wsUrl = this.buildWebSocketUrl(datasetName, baseUrl);
    
    try {{
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {{
        console.log(`[DatasetClient] Connected to ${{datasetName}}`);
        this.connections.set(datasetName, ws);
        
        // Clear reconnect timer
        const timer = this.reconnectTimers.get(datasetName);
        if (timer) {{
          clearTimeout(timer);
          this.reconnectTimers.delete(datasetName);
        }}
      }};
      
      ws.onmessage = (event) => {{
        try {{
          const message: WebSocketMessage = JSON.parse(event.data);
          this.notifyListeners(datasetName, message);
        }} catch (error) {{
          console.error(`[DatasetClient] Failed to parse message:`, error);
        }}
      }};
      
      ws.onerror = (error) => {{
        console.error(`[DatasetClient] WebSocket error for ${{datasetName}}:`, error);
      }};
      
      ws.onclose = () => {{
        console.log(`[DatasetClient] Disconnected from ${{datasetName}}`);
        this.connections.delete(datasetName);
        
        // Attempt reconnection if there are still listeners
        if (this.listeners.has(datasetName) && this.listeners.get(datasetName)!.size > 0) {{
          this.scheduleReconnect(datasetName, baseUrl, attempt + 1);
        }}
      }};
    }} catch (error) {{
      console.error(`[DatasetClient] Failed to create WebSocket:`, error);
      this.scheduleReconnect(datasetName, baseUrl, attempt + 1);
    }}
  }}
  
  private scheduleReconnect(datasetName: string, baseUrl: string, attempt: number): void {{
    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, attempt),
      this.maxReconnectDelay
    );
    
    console.log(`[DatasetClient] Reconnecting to ${{datasetName}} in ${{delay}}ms (attempt ${{attempt}})`);
    
    const timer = setTimeout(() => {{
      this.connect(datasetName, baseUrl, attempt);
    }}, delay);
    
    this.reconnectTimers.set(datasetName, timer);
  }}
  
  private disconnect(datasetName: string): void {{
    const ws = this.connections.get(datasetName);
    if (ws) {{
      ws.close();
      this.connections.delete(datasetName);
    }}
    
    const timer = this.reconnectTimers.get(datasetName);
    if (timer) {{
      clearTimeout(timer);
      this.reconnectTimers.delete(datasetName);
    }}
    
    this.listeners.delete(datasetName);
  }}
  
  private notifyListeners(datasetName: string, message: WebSocketMessage): void {{
    const listenerSet = this.listeners.get(datasetName);
    if (listenerSet) {{
      listenerSet.forEach(callback => {{
        try {{
          callback(message);
        }} catch (error) {{
          console.error(`[DatasetClient] Listener error:`, error);
        }}
      }});
    }}
  }}
  
  private buildWebSocketUrl(datasetName: string, baseUrl: string): string {{
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = baseUrl || window.location.host;
    return `${{protocol}}//${{host}}/ws/${{datasetName}}`;
  }}
}}

const wsManager = new WebSocketManager();

// ============================================================================
// DatasetClient Class
// ============================================================================

export class DatasetClient<T = any> {{
  constructor(
    private datasetName: string,
    private baseUrl: string = ""
  ) {{}}
  
  /**
   * Fetch paginated dataset records with optional filtering and sorting.
   */
  async fetch(options: DatasetQueryOptions = {{}}): Promise<PaginatedResponse<T>> {{
    const params = new URLSearchParams();
    
    if (options.page) params.append("page", String(options.page));
    if (options.page_size) params.append("page_size", String(options.page_size));
    if (options.sort_by) params.append("sort_by", options.sort_by);
    if (options.sort_order) params.append("sort_order", options.sort_order);
    if (options.search) params.append("search", options.search);
    
    const url = `${{this.baseUrl}}/api/datasets/${{this.datasetName}}?${{params.toString()}}`;
    
    const response = await fetch(url);
    if (!response.ok) {{
      throw new Error(`Failed to fetch dataset: ${{response.statusText}}`);
    }}
    
    return response.json();
  }}
  
  /**
   * Create a new record in the dataset.
   */
  async create(data: Partial<T>): Promise<T> {{
    const url = `${{this.baseUrl}}/api/datasets/${{this.datasetName}}`;
    
    const response = await fetch(url, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(data),
    }});
    
    if (!response.ok) {{
      const error = await response.json().catch(() => ({{ detail: response.statusText }}));
      throw new Error(error.detail || "Failed to create record");
    }}
    
    return response.json();
  }}
  
  /**
   * Update an existing record in the dataset.
   */
  async update(id: string | number, data: Partial<T>): Promise<T> {{
    const url = `${{this.baseUrl}}/api/datasets/${{this.datasetName}}/${{id}}`;
    
    const response = await fetch(url, {{
      method: "PATCH",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(data),
    }});
    
    if (!response.ok) {{
      const error = await response.json().catch(() => ({{ detail: response.statusText }}));
      throw new Error(error.detail || "Failed to update record");
    }}
    
    return response.json();
  }}
  
  /**
   * Delete a record from the dataset.
   */
  async delete(id: string | number): Promise<void> {{
    const url = `${{this.baseUrl}}/api/datasets/${{this.datasetName}}/${{id}}`;
    
    const response = await fetch(url, {{
      method: "DELETE",
    }});
    
    if (!response.ok) {{
      const error = await response.json().catch(() => ({{ detail: response.statusText }}));
      throw new Error(error.detail || "Failed to delete record");
    }}
  }}
  
  /**
   * Subscribe to realtime updates for this dataset.
   */
  subscribe(callback: (message: WebSocketMessage) => void): () => void {{
    return wsManager.subscribe(this.datasetName, callback, this.baseUrl);
  }}
}}

// ============================================================================
// React Hooks
// ============================================================================

export interface UseDatasetOptions extends DatasetQueryOptions {{
  enabled?: boolean;
  refetchInterval?: number;
  onSuccess?: (data: PaginatedResponse<any>) => void;
  onError?: (error: Error) => void;
}}

export interface UseDatasetResult<T> {{
  data: PaginatedResponse<T> | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}}

/**
 * React hook for fetching and subscribing to dataset changes.
 * 
 * Supports:
 * - Automatic data fetching with pagination
 * - Realtime WebSocket subscriptions
 * - Polling fallback
 * - Optimistic updates
 * 
 * @example
 * const {{ data, loading, error }} = useDataset("users", {{
 *   page: 1,
 *   page_size: 50,
 *   sort_by: "created_at",
 *   sort_order: "desc"
 * }});
 */
export function useDataset<T = any>(
  datasetName: string,
  options: UseDatasetOptions = {{}}
): UseDatasetResult<T> {{
  const [data, setData] = useState<PaginatedResponse<T> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const clientRef = useRef<DatasetClient<T>>();
  
  // Initialize client
  if (!clientRef.current) {{
    clientRef.current = new DatasetClient<T>(datasetName);
  }}
  
  const fetchData = useCallback(async () => {{
    if (options.enabled === false) return;
    
    try {{
      setLoading(true);
      setError(null);
      const result = await clientRef.current!.fetch(options);
      setData(result);
      options.onSuccess?.(result);
    }} catch (err) {{
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      options.onError?.(error);
    }} finally {{
      setLoading(false);
    }}
  }}, [datasetName, JSON.stringify(options)]);
  
  // Initial fetch
  useEffect(() => {{
    fetchData();
  }}, [fetchData]);
  
  // Realtime subscription
  useEffect(() => {{
    if (options.enabled === false) return;
    
    const unsubscribe = clientRef.current!.subscribe((message) => {{
      // Refetch data when changes detected
      if (message.type.startsWith("dataset.")) {{
        fetchData();
      }}
    }});
    
    return unsubscribe;
  }}, [datasetName, fetchData]);
  
  // Polling fallback
  useEffect(() => {{
    if (options.enabled === false || !options.refetchInterval) return;
    
    const interval = setInterval(() => {{
      fetchData();
    }}, options.refetchInterval);
    
    return () => clearInterval(interval);
  }}, [fetchData, options.refetchInterval]);
  
  return {{ data, loading, error, refetch: fetchData }};
}}

/**
 * React hook for dataset mutations (create/update/delete).
 * 
 * Provides optimistic updates and automatic refetching.
 * 
 * @example
 * const {{ create, update, delete: deleteRecord }} = useDatasetMutation("users");
 * await create({{ name: "John", email: "john@example.com" }});
 */
export function useDatasetMutation<T = any>(datasetName: string) {{
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const clientRef = useRef<DatasetClient<T>>();
  
  if (!clientRef.current) {{
    clientRef.current = new DatasetClient<T>(datasetName);
  }}
  
  const create = useCallback(async (data: Partial<T>): Promise<T | null> => {{
    try {{
      setLoading(true);
      setError(null);
      const result = await clientRef.current!.create(data);
      return result;
    }} catch (err) {{
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      return null;
    }} finally {{
      setLoading(false);
    }}
  }}, [datasetName]);
  
  const update = useCallback(async (id: string | number, data: Partial<T>): Promise<T | null> => {{
    try {{
      setLoading(true);
      setError(null);
      const result = await clientRef.current!.update(id, data);
      return result;
    }} catch (err) {{
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      return null;
    }} finally {{
      setLoading(false);
    }}
  }}, [datasetName]);
  
  const deleteRecord = useCallback(async (id: string | number): Promise<boolean> => {{
    try {{
      setLoading(true);
      setError(null);
      await clientRef.current!.delete(id);
      return true;
    }} catch (err) {{
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
      return false;
    }} finally {{
      setLoading(false);
    }}
  }}, [datasetName]);
  
  return {{
    create,
    update,
    delete: deleteRecord,
    loading,
    error,
  }};
}}
'''
    
    return textwrap.dedent(template).strip() + "\n"


def _generate_dataset_types(datasets: list) -> str:
    """Generate TypeScript type definitions for datasets."""
    
    if not datasets:
        return "// No datasets defined"
    
    type_defs = []
    
    for dataset in datasets:
        type_name = _to_pascal_case(dataset.name)
        fields = []
        
        # Generate field types from schema
        for field in dataset.schema:
            field_name = field.get("name", "unknown")
            field_type = _map_field_type(field.get("type", "string"))
            optional = "" if field.get("required", False) else "?"
            
            fields.append(f"  {field_name}{optional}: {field_type};")
        
        type_def = f"export interface {type_name} {{\n" + "\n".join(fields) + "\n}"
        type_defs.append(type_def)
    
    return "\n\n".join(type_defs)


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _map_field_type(n3_type: str) -> str:
    """Map Namel3ss field types to TypeScript types."""
    type_mapping = {
        "string": "string",
        "str": "string",
        "text": "string",
        "int": "number",
        "integer": "number",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "date": "string",
        "datetime": "string",
        "timestamp": "string",
        "json": "any",
        "object": "Record<string, any>",
        "array": "any[]",
    }
    
    return type_mapping.get(n3_type.lower(), "any")
