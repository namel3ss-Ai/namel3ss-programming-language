# Frontend Integration Modes for Namel3ss

## Overview

### Purpose

This document defines the architectural patterns for integrating frontend applications with the Namel3ss (N3) platform in production environments. It provides decision frameworks, implementation patterns, and operational guidance for teams building user-facing applications on top of N3's backend runtime.

**This is not a tutorial.** It assumes you understand modern frontend architectures (React, Next.js, SPAs), backend API design, and distributed systems concepts. If you need foundational N3 knowledge, start with the [Core Architecture](./ARCHITECTURE.md) and [Backend Integration](./BACKEND_INTEGRATION.md) guides.

### Intended Audience

- **Senior Frontend Engineers** architecting N3 integrations for existing React/Next.js/Vue applications
- **Platform Engineers** designing BFF (Backend-for-Frontend) layers that interact with N3
- **Technical Leads** making architectural decisions about N3 adoption strategies
- **DevOps/SRE Teams** planning deployment topologies and observability for N3-powered frontends

### When to Use This Document

Read this document when you need to:

1. **Integrate an existing frontend** with N3 workflows, agents, or datasets
2. **Choose an integration architecture** (remote service vs embedded runtime vs hybrid)
3. **Understand latency, security, and operational trade-offs** between integration modes
4. **Plan progressive N3 adoption** without rewriting your entire frontend
5. **Design API contracts** between your frontend and N3 backend services

**Context in N3 System Architecture:**

N3 runs as a backend runtime that executes declarative configurations (workflows, agents, datasets, tools). Your frontend needs a way to:
- Trigger N3 executions (start workflows, query datasets, invoke agents)
- Receive results (synchronous responses, streaming updates, webhooks)
- Handle errors and retries
- Maintain authentication and multi-tenant isolation

The integration mode determines **where N3 runs relative to your frontend** and **how data flows between them**. This impacts latency, deployment complexity, security boundaries, and team autonomy.

**Document Structure:**

- **Section 1-3**: Deep dives into each integration mode (Remote Service, Embedded Runtime, Hybrid)
- **Section 4**: Decision framework with trade-off analysis
- **Section 5**: Cross-cutting concerns (auth, long-running workflows, multi-tenancy)

---

## Integration Mode Comparison Table

The table below provides a high-level comparison of the three primary integration modes. Use this for initial architectural decisions, then refer to detailed sections for implementation specifics.

| Dimension | Remote Service | Embedded Runtime (BFF) | Hybrid |
|-----------|---------------|------------------------|--------|
| **N3 Location** | Separate backend service/cluster | In-process with frontend (Next.js API routes, BFF) | Both: remote backend + frontend-registered tools |
| **Communication** | HTTP/REST, WebSockets, gRPC | Function calls, in-process | Mixed: remote calls + local handlers |
| **Latency (typical)** | 50-200ms (network + processing) | 5-50ms (in-process, cold starts vary) | Variable: 5ms for local tools, 50-200ms for backend |
| **Deployment Model** | Independent: frontend deploys separately from N3 cluster | Coupled: N3 runtime ships with frontend/BFF | Independent backend + frontend handles coordination |
| **Scaling** | Scale N3 cluster independently | Scale frontend servers (must handle N3 memory/CPU) | Scale backend independently, frontend scales with tool load |
| **Team Boundaries** | Strong: frontend/backend teams work independently | Weak: shared deployment, tighter coordination | Medium: clear contracts but requires event coordination |
| **Auth Context** | Pass tokens in headers/cookies, N3 validates | Shared auth context (same process) | Mixed: backend validates, frontend inherits session |
| **Observability** | Distributed tracing across services | Single trace, easier debugging | Complex: traces span local + remote execution |
| **Cold Start Risk** | None (backend always warm) | High in serverless (Next.js edge, Lambda) | Medium: backend warm, BFF may have cold starts |
| **Polyglot Support** | Any frontend (React, Vue, mobile, CLI) | Tied to runtime (Node.js, Deno) | Flexible: any frontend can register tools via API |
| **Network Resilience** | Requires retry, timeout, circuit breaker logic | No network between frontend and N3 | Partial: local tools resilient, backend tools not |
| **State Management** | Frontend manages state, polls/subscribes to N3 | Shared memory possible (ephemeral) | Frontend manages state, coordinates via events |
| **Use Case** | Existing SPA, mobile app, multi-tenant SaaS | Low-latency requirements, Next.js/Remix apps | Complex workflows requiring both backend AI and frontend context |
| **Operational Complexity** | Medium: two services to deploy/monitor | Low-Medium: one deployment, but resource limits | High: event ordering, timeout policies, distributed state |

### Key Decision Factors

**Choose Remote Service if:**
- You have an existing frontend and want to add N3 features without refactoring
- Your teams have strong frontend/backend boundaries
- You need to support multiple frontend clients (web + mobile + CLI)
- You can tolerate 50-200ms latency for N3 operations
- You want independent scaling of frontend and N3 backend

**Choose Embedded Runtime if:**
- You're building a new Next.js or Remix app
- Sub-50ms latency is critical (e.g., real-time interactions, streaming)
- You want simplified deployment (single artifact)
- Your N3 workloads fit within serverless memory/CPU limits
- You're comfortable with tighter coupling between frontend and N3

**Choose Hybrid if:**
- You need backend N3 for heavy workloads (LLM agents, RAG pipelines) **and** frontend tools for UI-specific logic
- You're migrating from a legacy system and need progressive adoption
- Your workflows require client-side capabilities (browser APIs, local storage, user permissions)
- You can manage the complexity of distributed state machines
- You want to start with Remote Service but add local optimizations incrementally

### Anti-Patterns to Avoid

❌ **Don't embed N3 in the browser** (no client-side N3 runtime exists; security/isolation risks)  
❌ **Don't use Embedded Runtime for long-running workflows** (serverless timeouts, memory limits)  
❌ **Don't mix modes within a single feature** (pick one mode per workflow/feature for consistency)  
❌ **Don't skip distributed tracing** in Hybrid mode (debugging will be impossible)

---

## 1. Remote Service Mode

### Architecture

In Remote Service mode, N3 runs as a standalone backend service (or cluster of services), separate from your frontend. Your frontend communicates with N3 over the network using standard HTTP/WebSocket protocols.

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend Layer                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ React SPA    │  │ Next.js SSR  │  │ Mobile App   │      │
│  │              │  │              │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            │ HTTPS/WSS
                            │ (Auth: JWT/Session)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  API Gateway / Load Balancer                                 │
│  - Rate limiting                                             │
│  - Authentication                                            │
│  - Request routing                                           │
│  - CORS headers                                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  N3 Backend Service (Cluster)                                │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ N3 Runtime       │  │ N3 Runtime       │  (scaled)       │
│  │ - Workflows      │  │ - Workflows      │                │
│  │ - Agents         │  │ - Agents         │                │
│  │ - Datasets       │  │ - Datasets       │                │
│  │ - Tools          │  │ - Tools          │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                           │
└───────────┼─────────────────────┼───────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Persistence & External Services                             │
│  - PostgreSQL (datasets)                                     │
│  - Redis (execution state)                                   │
│  - Vector DB (RAG)                                           │
│  - LLM APIs (OpenAI, Anthropic)                             │
└─────────────────────────────────────────────────────────────┘
```

### Request/Response Flow

**Synchronous Request (short-lived workflow):**

```
1. Frontend → POST /api/v1/workflows/{workflow_id}/execute
   Headers: Authorization: Bearer <jwt>, X-Trace-ID: <uuid>
   Body: { "input": { ...params }, "config": { "timeout": 30000 } }

2. API Gateway → Validate JWT, extract tenant_id, route to N3 instance

3. N3 Runtime → Execute workflow, call tools, return result

4. N3 Backend → Response (200 OK)
   Body: { 
     "execution_id": "exec_abc123",
     "status": "completed",
     "result": { ...output },
     "duration_ms": 1240,
     "trace_url": "https://traces.your-backend.com/exec_abc123"
   }

5. Frontend → Render result or handle error
```

**Asynchronous Request (long-running workflow):**

```
1. Frontend → POST /api/v1/workflows/{workflow_id}/execute
   Body: { "input": { ...params }, "async": true }

2. N3 Backend → Immediate response (202 Accepted)
   Body: { 
     "execution_id": "exec_xyz789",
     "status": "running",
     "status_url": "/api/v1/executions/exec_xyz789"
   }

3. Frontend → Poll or subscribe to status updates:
   
   Option A (Polling):
     GET /api/v1/executions/exec_xyz789/status (every 2-5 seconds)
   
   Option B (WebSocket):
     WSS /api/v1/executions/exec_xyz789/stream
     → Receives events: { "type": "progress", "data": {...} }
                        { "type": "completed", "result": {...} }
   
   Option C (Server-Sent Events):
     GET /api/v1/executions/exec_xyz789/events
     → text/event-stream with progress updates

4. N3 Backend → Workflow completes, stores result

5. Frontend → Receives final status, displays result
```

### Authentication & Session Handling

**Token Propagation:**

```typescript
// Frontend: Attach auth token to every N3 request
const executeWorkflow = async (workflowId: string, input: unknown) => {
  const token = await getAuthToken(); // From your auth provider
  
  const response = await fetch(`${N3_API_URL}/workflows/${workflowId}/execute`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'X-Tenant-ID': getCurrentTenant(), // Multi-tenancy
      'X-Trace-ID': generateTraceId(),   // Observability
    },
    body: JSON.stringify({ input }),
  });
  
  if (!response.ok) {
    throw new N3Error(await response.json());
  }
  
  return response.json();
};
```

**Backend Validation:**

The API Gateway or N3 service validates the JWT on every request:
- Extracts `user_id`, `tenant_id`, `permissions` from token claims
- Enforces authorization policies (RBAC, ABAC)
- Passes validated context to N3 runtime as execution metadata

**Session Management:**

- **Stateless:** JWT contains all necessary claims, no server-side session storage
- **Refresh Tokens:** Frontend handles token refresh transparently (401 → refresh → retry)
- **CSRF Protection:** Use SameSite cookies + CSRF tokens for browser-based SPAs

### Implementation Patterns

#### Pattern 1: REST API Integration

Use for most CRUD operations and synchronous workflows.

**TypeScript Client Example:**

```typescript
// n3-client.ts
export class N3Client {
  constructor(
    private baseUrl: string,
    private getToken: () => Promise<string>
  ) {}

  async executeWorkflow<TInput, TOutput>(
    workflowId: string,
    input: TInput,
    options?: { timeout?: number; async?: boolean }
  ): Promise<N3Execution<TOutput>> {
    const token = await this.getToken();
    
    const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/execute`, {
      method: 'POST',
      headers: this.buildHeaders(token),
      body: JSON.stringify({ input, ...options }),
      signal: AbortSignal.timeout(options?.timeout || 30000),
    });

    if (!response.ok) {
      throw await this.parseError(response);
    }

    return response.json();
  }

  async getExecutionStatus<TOutput>(
    executionId: string
  ): Promise<N3Execution<TOutput>> {
    const token = await this.getToken();
    
    const response = await fetch(`${this.baseUrl}/executions/${executionId}`, {
      headers: this.buildHeaders(token),
    });

    return response.json();
  }

  private buildHeaders(token: string): HeadersInit {
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'X-Tenant-ID': getCurrentTenant(),
      'X-Trace-ID': generateTraceId(),
    };
  }

  private async parseError(response: Response): Promise<Error> {
    const body = await response.json();
    return new N3Error(body.error, body.details, response.status);
  }
}
```

#### Pattern 2: WebSocket Subscriptions

Use for long-running workflows requiring real-time updates.

**React Hook Example:**

```typescript
// useN3Execution.ts
export function useN3Execution<TOutput>(executionId: string | null) {
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [result, setResult] = useState<TOutput | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!executionId) return;

    const ws = new WebSocket(`${N3_WS_URL}/executions/${executionId}/stream`);
    
    ws.onopen = () => {
      // Send auth token after connection
      ws.send(JSON.stringify({ type: 'auth', token: getAuthToken() }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'status':
          setStatus(message.status);
          break;
        case 'progress':
          // Update UI with progress (optional)
          break;
        case 'completed':
          setStatus('completed');
          setResult(message.result);
          ws.close();
          break;
        case 'failed':
          setStatus('failed');
          setError(new Error(message.error));
          ws.close();
          break;
      }
    };

    ws.onerror = () => {
      setError(new Error('WebSocket connection failed'));
    };

    return () => {
      ws.close();
    };
  }, [executionId]);

  return { status, result, error };
}
```

#### Pattern 3: Server-Sent Events (SSE)

Alternative to WebSockets for server-to-client streaming (simpler, HTTP-based).

```typescript
// useN3ExecutionSSE.ts
export function useN3ExecutionSSE<TOutput>(executionId: string | null) {
  const [events, setEvents] = useState<N3Event[]>([]);
  const [result, setResult] = useState<TOutput | null>(null);

  useEffect(() => {
    if (!executionId) return;

    const eventSource = new EventSource(
      `${N3_API_URL}/executions/${executionId}/events`,
      { withCredentials: true } // Include cookies for auth
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents(prev => [...prev, data]);
      
      if (data.type === 'completed') {
        setResult(data.result);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [executionId]);

  return { events, result };
}
```

### Pros & Cons

#### Strengths

✅ **Clear Separation of Concerns**
- Frontend and backend teams can work independently
- Deploy, scale, and version services separately
- Change N3 backend without touching frontend code (as long as API contracts are stable)

✅ **Polyglot Frontend Support**
- Same N3 backend works with React, Vue, Angular, mobile apps, CLI tools
- No runtime constraints on frontend (can use any framework, browser, or platform)

✅ **Independent Scaling**
- Scale N3 cluster based on workflow load
- Scale frontend based on user traffic
- Optimize each tier independently (CPU for N3, memory for frontend)

✅ **Security Isolation**
- N3 backend runs in isolated VPC/network
- Secrets (API keys, DB credentials) never exposed to frontend
- API Gateway enforces rate limiting, auth, and quotas per tenant

✅ **Mature Operational Model**
- Standard microservices observability (Prometheus, Grafana, Datadog)
- Well-understood deployment patterns (Kubernetes, ECS, Cloud Run)
- Circuit breakers, retries, and failover strategies are proven

#### Weaknesses

❌ **Network Latency**
- Minimum 50-200ms round-trip for each N3 call
- Not suitable for sub-50ms interactive features (e.g., autocomplete, real-time collaboration)
- Cumulative latency in multi-step workflows

❌ **Coordination Overhead**
- Requires API versioning strategy between frontend and backend
- Breaking changes require coordinated deploys or backward compatibility layers
- More moving parts to monitor and debug

❌ **Stateful Workflows Are Complex**
- Frontend must manage polling, reconnection, and state persistence
- No shared memory between frontend and N3 (state lives in Redis/DB)
- Resuming interrupted workflows requires careful design

❌ **Requires Distributed Tracing**
- Errors span multiple services (frontend → gateway → N3 → external APIs)
- Debugging requires correlation IDs and centralized logging
- Operational complexity increases with service count

### When to Use

**Ideal for:**

1. **Existing SPA/Mobile Apps** adding N3-powered features (search, recommendations, agents)
2. **Multi-Tenant SaaS** where N3 cluster serves many organizations
3. **Regulated Industries** requiring strict network isolation (healthcare, finance)
4. **Polyglot Teams** where different teams use different frontend stacks
5. **Long-Running Workflows** (batch jobs, reports, async processing)

**Not recommended for:**

1. **Real-Time Interactive Features** (sub-50ms latency required)
2. **Offline-First Apps** (no network = no N3 access)
3. **High-Volume, Low-Latency APIs** (e.g., autocomplete with 10K req/sec)

### Operational Considerations

#### API Versioning Strategy

Use semantic versioning for N3 APIs:

```
/api/v1/workflows/{id}/execute  → Stable, backward-compatible
/api/v2/workflows/{id}/execute  → Breaking changes, parallel deployment
```

**Migration Path:**
1. Deploy v2 alongside v1
2. Add deprecation warnings to v1 responses
3. Migrate frontend clients incrementally
4. Sunset v1 after 6-12 months

#### Rate Limiting & Quotas

Enforce per-tenant limits at API Gateway:

```yaml
rate_limits:
  free_tier:
    requests_per_minute: 60
    concurrent_executions: 5
  pro_tier:
    requests_per_minute: 600
    concurrent_executions: 50
  enterprise:
    requests_per_minute: unlimited
    concurrent_executions: 500
```

Return `429 Too Many Requests` with `Retry-After` header when limits are exceeded.

#### CORS & Security Headers

Configure API Gateway to return:

```http
Access-Control-Allow-Origin: https://app.your-domain.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, X-Trace-ID, X-Tenant-ID
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400

Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
```

Never use `Access-Control-Allow-Origin: *` in production with credentials.

#### Monitoring & Distributed Tracing

**Required Metrics:**
- Request rate, error rate, latency (p50, p95, p99) per endpoint
- Execution duration per workflow/agent
- Queue depth for async executions
- WebSocket connection count and duration

**Distributed Tracing:**
- Propagate `X-Trace-ID` from frontend through all services
- Use OpenTelemetry or AWS X-Ray for cross-service traces
- Include trace URL in API responses for debugging

**Example Response:**

```json
{
  "execution_id": "exec_abc123",
  "status": "completed",
  "result": { ... },
  "duration_ms": 1240,
  "trace_url": "https://traces.your-backend.com/exec_abc123"
}
```

Frontend can display trace URL in dev tools or error modals for support teams.

---

## 2. Embedded Runtime Mode (BFF/Edge)

### Architecture

In Embedded Runtime mode, the N3 runtime runs in-process with your frontend server layer (Next.js API routes, Remix loaders, or a dedicated BFF service). There is no network hop between your frontend logic and N3—they share the same process, memory, and execution context.

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│  Browser / Client Layer                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  React/Vue Components                                 │   │
│  │  - Forms, dashboards, interactive UI                  │   │
│  │  - Client-side state management                       │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │
                      │ HTTPS (SSR/API Routes)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend Server (Next.js / Remix / BFF)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Server-Side Code (Same Process)                      │   │
│  │  ┌────────────────┐    ┌────────────────┐           │   │
│  │  │ API Routes     │───▶│ N3 Runtime     │           │   │
│  │  │ /api/execute   │    │ (Embedded)     │           │   │
│  │  │                │◀───│ - Workflows    │           │   │
│  │  │ Server Actions │    │ - Agents       │           │   │
│  │  │ Loaders        │    │ - Datasets     │           │   │
│  │  └────────────────┘    └────────┬───────┘           │   │
│  │                                  │                    │   │
│  │  Auth Context ─────────────────▶│ (Shared Session)  │   │
│  └──────────────────────────────────┼────────────────────┘   │
│                                     │                        │
└─────────────────────────────────────┼────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │  External Services       │
                        │  - PostgreSQL (datasets) │
                        │  - Redis (optional)      │
                        │  - LLM APIs              │
                        └──────────────────────────┘
```

**Key Difference from Remote Service:**
- No API Gateway or separate N3 cluster
- Function calls instead of HTTP requests
- Shared memory and auth context
- Deployed as a single artifact

### In-Process Execution Flow

**Next.js API Route Example:**

```typescript
// app/api/workflows/[workflowId]/execute/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { getN3Runtime } from '@/lib/n3';
import { getServerSession } from 'next-auth';

export async function POST(
  request: NextRequest,
  { params }: { params: { workflowId: string } }
) {
  // 1. Validate session (in-process, no network call)
  const session = await getServerSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // 2. Parse input
  const { input } = await request.json();

  // 3. Execute N3 workflow (in-process, no network)
  const runtime = getN3Runtime();
  
  try {
    const result = await runtime.executeWorkflow(params.workflowId, {
      input,
      context: {
        user_id: session.user.id,
        tenant_id: session.user.tenantId,
        trace_id: request.headers.get('x-trace-id'),
      },
    });

    // 4. Return result (5-50ms total latency)
    return NextResponse.json({
      execution_id: result.id,
      status: 'completed',
      result: result.output,
      duration_ms: result.duration,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error.message, trace_id: error.traceId },
      { status: 500 }
    );
  }
}
```

**React Server Component Example:**

```typescript
// app/dashboard/page.tsx (Server Component)
import { getN3Runtime } from '@/lib/n3';
import { getServerSession } from 'next-auth';
import { DashboardView } from '@/components/DashboardView';

export default async function DashboardPage() {
  const session = await getServerSession();
  const runtime = getN3Runtime();

  // Execute N3 workflow directly in Server Component
  const dashboardData = await runtime.executeWorkflow('dashboard_data', {
    input: { user_id: session.user.id },
    context: { tenant_id: session.user.tenantId },
  });

  // Render with data (no client-side fetch needed)
  return <DashboardView data={dashboardData.result} />;
}
```

### Implementation Patterns

#### Pattern 1: N3 Runtime Singleton

Initialize N3 runtime once and reuse across requests:

```typescript
// lib/n3.ts
import { N3Runtime } from '@namel3ss/runtime';

let runtimeInstance: N3Runtime | null = null;

export function getN3Runtime(): N3Runtime {
  if (!runtimeInstance) {
    runtimeInstance = new N3Runtime({
      configPath: process.env.N3_CONFIG_PATH,
      databaseUrl: process.env.DATABASE_URL,
      enableTracing: process.env.NODE_ENV === 'production',
      maxConcurrentExecutions: 10, // Per server instance
    });
  }
  return runtimeInstance;
}

// Cleanup on shutdown (for graceful deploys)
process.on('SIGTERM', async () => {
  if (runtimeInstance) {
    await runtimeInstance.shutdown();
  }
});
```

#### Pattern 2: Server Actions (Next.js 14+)

Use Server Actions for form submissions and mutations:

```typescript
// app/actions/workflow-actions.ts
'use server';

import { getN3Runtime } from '@/lib/n3';
import { getServerSession } from 'next-auth';
import { revalidatePath } from 'next/cache';

export async function executeWorkflowAction(
  workflowId: string,
  formData: FormData
) {
  const session = await getServerSession();
  if (!session) throw new Error('Unauthorized');

  const runtime = getN3Runtime();
  const input = Object.fromEntries(formData);

  const result = await runtime.executeWorkflow(workflowId, {
    input,
    context: { user_id: session.user.id },
  });

  // Revalidate cache after mutation
  revalidatePath('/dashboard');

  return { success: true, result: result.output };
}
```

```typescript
// components/WorkflowForm.tsx (Client Component)
'use client';

import { executeWorkflowAction } from '@/app/actions/workflow-actions';
import { useFormStatus } from 'react-dom';

export function WorkflowForm() {
  return (
    <form action={(formData) => executeWorkflowAction('process_data', formData)}>
      <input name="field1" required />
      <input name="field2" required />
      <SubmitButton />
    </form>
  );
}

function SubmitButton() {
  const { pending } = useFormStatus();
  return <button disabled={pending}>Submit</button>;
}
```

#### Pattern 3: Remix Loaders

Load N3 data in Remix loaders:

```typescript
// app/routes/dashboard.tsx
import { json, LoaderFunction } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';
import { getN3Runtime } from '~/lib/n3.server';
import { requireAuth } from '~/lib/auth.server';

export const loader: LoaderFunction = async ({ request }) => {
  const session = await requireAuth(request);
  const runtime = getN3Runtime();

  const data = await runtime.executeWorkflow('dashboard_summary', {
    input: { user_id: session.userId },
    context: { tenant_id: session.tenantId },
  });

  return json({ dashboardData: data.result });
};

export default function Dashboard() {
  const { dashboardData } = useLoaderData<typeof loader>();
  return <div>{/* Render dashboard with data */}</div>;
}
```

#### Pattern 4: Edge Runtime Considerations

When deploying to edge runtimes (Vercel Edge, Cloudflare Workers), N3 must fit within constraints:

```typescript
// middleware.ts (Edge Runtime)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Edge-compatible N3 runtime (if supported)
// Note: Full N3 may not run on edge due to native dependencies
export const config = {
  runtime: 'edge', // or 'nodejs' for full N3
};

export async function middleware(request: NextRequest) {
  // Lightweight N3 operations only (dataset queries, simple workflows)
  // Heavy operations (LLM agents, RAG) should stay in Node.js runtime
  
  const response = NextResponse.next();
  return response;
}
```

**Constraints:**
- Edge runtimes have limited CPU/memory (128MB-512MB)
- No native Node.js modules (some N3 features may not work)
- Short execution time limits (10-30 seconds)
- Use edge for simple workflows, Node.js runtime for complex ones

### Pros & Cons

#### Strengths

✅ **Low Latency**
- No network overhead: 5-50ms typical latency (vs 50-200ms for remote)
- Synchronous function calls instead of HTTP requests
- Perfect for interactive features (autocomplete, real-time validation)

✅ **Simplified Deployment**
- Single artifact to build and deploy
- No service coordination or API versioning between frontend and N3
- Easier to reason about in development (single codebase)

✅ **Shared Auth Context**
- Session data immediately available to N3 (no token parsing)
- No risk of token expiration mid-execution
- Simplified multi-tenancy (tenant context flows naturally)

✅ **Easier Local Development**
- Run entire stack with `npm run dev` (no Docker Compose for N3 backend)
- Hot reload works for both frontend and N3 changes
- Single debugger session covers frontend + N3

✅ **Reduced Operational Complexity**
- One deployment pipeline instead of two
- Single set of logs to monitor
- No distributed tracing complexity (single-process traces)

#### Weaknesses

❌ **Tight Coupling**
- Frontend and N3 must deploy together (no independent versioning)
- N3 changes can break frontend build or require coordinated releases
- Harder to scale frontend and N3 independently

❌ **Resource Constraints**
- N3 competes with frontend for CPU/memory on same server
- Serverless platforms have strict limits (512MB-3GB memory, 10-300s timeout)
- Long-running workflows will hit timeouts (use Remote Service for those)

❌ **Cold Start Latency**
- Serverless deployments: first request after idle can take 1-10 seconds
- N3 runtime initialization adds to cold start time
- Mitigate with provisioned concurrency (increases cost)

❌ **Limited Polyglot Support**
- Tied to Node.js/Deno runtime (no mobile, CLI, or other languages)
- Can't serve multiple frontend clients (web + mobile) from same N3 instance
- If you add a mobile app later, you'll need Remote Service anyway

❌ **Harder to Isolate Failures**
- N3 error can crash entire frontend server (requires robust error handling)
- Memory leaks in N3 affect frontend stability
- No circuit breaker between frontend and N3 (they're the same process)

### When to Use

**Ideal for:**

1. **New Next.js/Remix Apps** built from scratch with N3
2. **Low-Latency Features** like autocomplete, real-time search, interactive agents
3. **Small-to-Medium Workloads** fitting within serverless memory/CPU limits
4. **Single-Platform Apps** (web-only, no mobile or CLI clients needed)
5. **Simplified DevOps** where unified deployment is a priority

**Not recommended for:**

1. **Long-Running Workflows** (>30 seconds) that exceed serverless timeouts
2. **Memory-Intensive Workflows** (large RAG contexts, video processing)
3. **Multi-Platform Products** needing to support web + mobile + CLI
4. **Independent Team Boundaries** (separate frontend/backend teams)
5. **Existing Apps** with established frontend/backend separation

### Operational Considerations

#### Memory & CPU Limits

**Serverless Platforms:**

| Platform | Max Memory | Max Timeout | Max Request Size |
|----------|-----------|-------------|------------------|
| Vercel (Hobby) | 1GB | 10s | 4.5MB |
| Vercel (Pro) | 3GB | 60s | 4.5MB |
| Netlify | 1GB | 10s | 6MB |
| AWS Lambda | 10GB | 15min | 6MB |
| Cloudflare Workers | 128MB | 30s CPU | 100MB |

**Recommendations:**
- Profile N3 workflows to measure memory usage
- Use provisioned concurrency for consistent performance (eliminates cold starts)
- Set resource limits in N3 config to prevent runaway executions

```typescript
// lib/n3.ts
const runtime = new N3Runtime({
  maxMemoryMB: 512,        // Fail if workflow exceeds 512MB
  maxExecutionTimeMs: 25000, // Timeout before platform limit (30s)
  enableMemoryProfiling: process.env.NODE_ENV === 'development',
});
```

#### Cold Start Mitigation

**Strategies:**

1. **Provisioned Concurrency** (Vercel Pro, AWS Lambda)
   - Keep 1-5 instances warm at all times
   - Adds cost but eliminates cold starts for most requests

2. **Lazy Initialization**
   - Defer expensive N3 setup until first workflow execution
   - Cache compiled workflows in memory

3. **Edge + Origin Hybrid**
   - Use edge runtime for routing and simple queries
   - Delegate heavy N3 workflows to origin (Node.js runtime)

4. **Warm-Up Requests**
   - Send periodic synthetic requests to keep functions warm
   - Use cron job or monitoring tool

```typescript
// app/api/warmup/route.ts
export async function GET() {
  // Trigger N3 initialization without doing real work
  const runtime = getN3Runtime();
  await runtime.healthCheck();
  return new Response('OK');
}

// Cron: curl https://your-app.com/api/warmup every 5 minutes
```

#### Logging & Observability

**Single-Process Tracing:**

Since frontend and N3 run in the same process, use structured logging:

```typescript
// lib/logger.ts
import { Logger } from 'pino';

export const logger = Logger({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
});

// In API route
logger.info({
  event: 'workflow_start',
  workflow_id: workflowId,
  user_id: session.user.id,
  trace_id: traceId,
});

const result = await runtime.executeWorkflow(workflowId, input);

logger.info({
  event: 'workflow_complete',
  workflow_id: workflowId,
  duration_ms: result.duration,
  trace_id: traceId,
});
```

**N3 Runtime Integration:**

```typescript
const runtime = new N3Runtime({
  logger: logger.child({ component: 'n3_runtime' }),
  tracing: {
    enabled: true,
    exporter: 'console', // or 'otlp' for production
  },
});
```

#### Graceful Shutdown

Handle in-flight N3 executions during deployment:

```typescript
// lib/n3.ts
let isShuttingDown = false;

process.on('SIGTERM', async () => {
  console.log('SIGTERM received, starting graceful shutdown...');
  isShuttingDown = true;

  // Stop accepting new requests
  // Wait for in-flight executions to complete (up to 30s)
  if (runtimeInstance) {
    await runtimeInstance.shutdown({ timeout: 30000 });
  }

  process.exit(0);
});

// In API route
export async function POST(request: NextRequest) {
  if (isShuttingDown) {
    return NextResponse.json(
      { error: 'Service shutting down' },
      { status: 503, headers: { 'Retry-After': '10' } }
    );
  }
  // ... execute workflow
}
```

---

## Section 3: Hybrid Mode (Remote Backend + Frontend-Registered Tools)

### Architecture

Hybrid mode combines a **remote N3 backend** with **frontend-registered tools** that the backend can invoke during workflow execution. The backend runs N3 workflows in isolation, but when it encounters a tool the frontend registered (e.g., `ui.showModal`, `ui.requestApproval`, `ui.uploadFile`), it sends an event to the frontend and waits for the response.

**Key Characteristics:**
- Backend runs N3 workflows independently (compute isolation)
- Frontend registers "tools" the backend can call (bidirectional RPC)
- Event-driven: backend sends `tool_call_requested` → frontend responds → backend resumes
- Distributed state: workflow state lives in backend, UI state in frontend

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Hybrid Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐                    ┌──────────────────────┐  │
│  │   React/Vue/     │  WebSocket/SSE     │   N3 Backend         │  │
│  │   Svelte App     │◄──────────────────►│   (FastAPI)          │  │
│  └──────────────────┘                    └──────────────────────┘  │
│         │                                           │               │
│         │ 1. Register tools                         │               │
│         │   {id: "ui.showModal", ...}              │               │
│         ├──────────────────────────────────────────►│               │
│         │                                           │               │
│         │ 2. Execute workflow                       │               │
│         │   POST /workflows/123/execute            │               │
│         ├──────────────────────────────────────────►│               │
│         │                                           │               │
│         │                                           │ 3. N3 hits    │
│         │                                           │    tool call  │
│         │                                           │               │
│         │ 4. tool_call_requested event              │               │
│         │◄──────────────────────────────────────────┤               │
│         │   {tool: "ui.showModal", args: {...}}   │               │
│         │                                           │               │
│         │ 5. Frontend executes tool                 │               │
│         │    (shows modal, waits for user)         │               │
│         │                                           │               │
│         │ 6. Send tool result                       │               │
│         │   {result: "approved"}                   │               │
│         ├──────────────────────────────────────────►│               │
│         │                                           │               │
│         │                                           │ 7. N3 resumes │
│         │                                           │    execution  │
│         │                                           │               │
│         │ 8. workflow_completed                     │               │
│         │◄──────────────────────────────────────────┤               │
│         │                                           │               │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Bidirectional Event Flow

**Tool Registration (Startup):**
```typescript
// Frontend registers tools the backend can invoke
const ws = new WebSocket('wss://api.example.com/workflows/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'register_tools',
    tools: [
      {
        id: 'ui.showModal',
        schema: {
          title: { type: 'string' },
          message: { type: 'string' },
          actions: { type: 'array', items: { type: 'string' } }
        }
      },
      {
        id: 'ui.uploadFile',
        schema: {
          accept: { type: 'string' },
          maxSize: { type: 'number' }
        }
      },
      {
        id: 'ui.requestApproval',
        schema: {
          approvers: { type: 'array', items: { type: 'string' } },
          timeout_ms: { type: 'number' }
        }
      }
    ]
  }));
};
```

**Tool Invocation (Runtime):**
```typescript
// Backend sends tool_call_requested → Frontend executes → Backend resumes
ws.onmessage = async (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'tool_call_requested') {
    const { call_id, tool, args, execution_id } = msg;

    try {
      let result;
      
      // Route to appropriate UI handler
      switch (tool) {
        case 'ui.showModal':
          result = await showModalHandler(args);
          break;
        case 'ui.uploadFile':
          result = await uploadFileHandler(args);
          break;
        case 'ui.requestApproval':
          result = await requestApprovalHandler(args);
          break;
        default:
          throw new Error(`Unknown tool: ${tool}`);
      }

      // Send result back to backend
      ws.send(JSON.stringify({
        type: 'tool_call_response',
        call_id,
        execution_id,
        result,
        error: null
      }));

    } catch (error) {
      ws.send(JSON.stringify({
        type: 'tool_call_response',
        call_id,
        execution_id,
        result: null,
        error: error.message
      }));
    }
  }
};

// Example: showModal implementation
async function showModalHandler(args: {
  title: string;
  message: string;
  actions: string[];
}): Promise<string> {
  return new Promise((resolve) => {
    const modal = createModal({
      title: args.title,
      message: args.message,
      actions: args.actions,
      onAction: (action) => {
        modal.close();
        resolve(action);
      }
    });
    modal.show();
  });
}
```

---

### Implementation Patterns

#### Pattern 1: Approval Workflows with UI Gates

**Use Case:** Multi-step workflow that requires human approval before proceeding (e.g., payment processing, content moderation).

**N3 Workflow:**
```python
# approval_workflow.n3
workflow ApprovalWorkflow {
  input: {
    "amount": 5000,
    "vendor": "Acme Corp"
  }
  
  steps: [
    # Step 1: Validate request
    {
      "id": "validate",
      "tool": "accounting.validate_invoice",
      "args": {"amount": input.amount, "vendor": input.vendor}
    },
    
    # Step 2: Request approval from finance team
    {
      "id": "request_approval",
      "tool": "ui.requestApproval",
      "args": {
        "approvers": ["finance@example.com"],
        "timeout_ms": 300000,  # 5 minutes
        "context": {
          "amount": input.amount,
          "vendor": input.vendor,
          "invoice_id": steps.validate.invoice_id
        }
      }
    },
    
    # Step 3: Process payment (only if approved)
    {
      "id": "process_payment",
      "tool": "payments.process",
      "args": {"invoice_id": steps.validate.invoice_id},
      "condition": "steps.request_approval.result == 'approved'"
    }
  ]
}
```

**Frontend Implementation:**
```typescript
// Approval handler with timeout
async function requestApprovalHandler(args: {
  approvers: string[];
  timeout_ms: number;
  context: Record<string, any>;
}): Promise<string> {
  const timeoutPromise = new Promise<string>((_, reject) =>
    setTimeout(() => reject(new Error('Approval timeout')), args.timeout_ms)
  );

  const approvalPromise = new Promise<string>((resolve) => {
    // Show approval modal
    const modal = createApprovalModal({
      amount: args.context.amount,
      vendor: args.context.vendor,
      approvers: args.approvers,
      onApprove: () => {
        logApprovalEvent({ action: 'approved', ...args.context });
        modal.close();
        resolve('approved');
      },
      onReject: () => {
        logApprovalEvent({ action: 'rejected', ...args.context });
        modal.close();
        resolve('rejected');
      }
    });
    
    modal.show();
  });

  return Promise.race([approvalPromise, timeoutPromise]);
}
```

---

#### Pattern 2: File Upload Workflows

**Use Case:** Backend workflow needs user to upload a file (e.g., document processing, image analysis).

**N3 Workflow:**
```python
# document_processing.n3
workflow DocumentProcessing {
  steps: [
    # Step 1: Request file upload from user
    {
      "id": "request_upload",
      "tool": "ui.uploadFile",
      "args": {
        "accept": ".pdf,.docx",
        "maxSize": 10485760  # 10 MB
      }
    },
    
    # Step 2: Upload file to storage
    {
      "id": "store_file",
      "tool": "storage.upload",
      "args": {
        "file_data": steps.request_upload.file_data,
        "file_name": steps.request_upload.file_name
      }
    },
    
    # Step 3: Process document
    {
      "id": "process",
      "tool": "ocr.extract_text",
      "args": {"file_url": steps.store_file.url}
    }
  ]
}
```

**Frontend Implementation:**
```typescript
async function uploadFileHandler(args: {
  accept: string;
  maxSize: number;
}): Promise<{ file_name: string; file_data: string; mime_type: string }> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = args.accept;
    
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) {
        reject(new Error('No file selected'));
        return;
      }
      
      if (file.size > args.maxSize) {
        reject(new Error(`File too large: ${file.size} > ${args.maxSize}`));
        return;
      }
      
      // Convert to base64 (or upload directly to S3 and return URL)
      const reader = new FileReader();
      reader.onload = () => {
        resolve({
          file_name: file.name,
          file_data: reader.result as string,
          mime_type: file.type
        });
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    };
    
    input.click();
  });
}
```

---

#### Pattern 3: Real-Time Progress Updates

**Use Case:** Long-running workflow with UI progress indicators (e.g., data import, batch processing).

**N3 Workflow:**
```python
# batch_import.n3
workflow BatchImport {
  input: {"file_ids": [1, 2, 3, 4, 5]}
  
  steps: [
    {
      "id": "process_files",
      "tool": "batch.process",
      "args": {"file_ids": input.file_ids},
      "on_progress": "ui.updateProgress"  # Special hook
    }
  ]
}
```

**Frontend Implementation:**
```typescript
// Backend sends progress events during execution
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  if (msg.type === 'progress_update') {
    const { execution_id, step_id, progress } = msg;
    
    // Update UI progress bar
    updateProgressBar(execution_id, {
      current: progress.current,
      total: progress.total,
      percent: (progress.current / progress.total) * 100,
      message: progress.message
    });
  }
};

// Backend sends progress updates
// (Backend implementation)
async def process_batch(file_ids: list[int], progress_callback):
    total = len(file_ids)
    for i, file_id in enumerate(file_ids):
        await process_file(file_id)
        progress_callback({
            "current": i + 1,
            "total": total,
            "message": f"Processed file {file_id}"
        })
```

---

### Pros & Cons

**Strengths:**
1. **Best of both worlds**: Remote compute isolation + UI flexibility
2. **Long-running workflows**: Backend can run workflows for hours while frontend reconnects
3. **Rich UI interactions**: Native modals, file uploads, approval flows without backend UI rendering
4. **Multi-platform**: Same backend serves web, mobile, desktop with platform-specific UI tools
5. **Scalability**: Backend scales independently, frontend handles UI-specific logic
6. **Auditability**: All tool calls logged centrally (who approved what, when)

**Weaknesses:**
1. **Complexity**: Requires WebSocket/SSE infrastructure, event-driven state management
2. **Timeout handling**: Tool calls can timeout if user navigates away or loses connection
3. **Debugging**: Distributed tracing required to follow workflow → frontend → backend flow
4. **Tool versioning**: Frontend and backend must agree on tool schemas (schema registry recommended)
5. **Testing**: Requires mocking bidirectional events in tests

---

### When to Use Hybrid Mode

**Use this mode when:**
- Workflows require **human-in-the-loop** interactions (approvals, file uploads, confirmations)
- You need **UI gates** in long-running backend processes (e.g., payment workflows, content moderation)
- Backend workflows must access **frontend-only capabilities** (camera, geolocation, browser storage)
- You want **multi-platform support** (web, mobile, desktop) with platform-specific UI tools
- Workflows involve **sensitive operations** that require explicit user consent (delete data, charge card)

**Do NOT use this mode when:**
- All operations can be completed without user interaction (pure automation)
- Latency is critical and UI interactions would slow down the workflow
- You can't guarantee WebSocket/SSE connectivity (unreliable networks)
- Workflow logic is simple enough for Remote Service Mode (avoid over-engineering)

---

### Operational Considerations

#### Event Ordering & Idempotency

**Challenge:** Tool calls may arrive out-of-order or be retried due to network issues.

**Solution:**
```typescript
// Track processed tool calls to prevent duplicate execution
const processedCalls = new Set<string>();

ws.onmessage = async (event) => {
  const msg = JSON.parse(event.data);
  
  if (msg.type === 'tool_call_requested') {
    const { call_id } = msg;
    
    // Idempotency check
    if (processedCalls.has(call_id)) {
      console.warn(`Duplicate tool call: ${call_id}`);
      return;
    }
    processedCalls.add(call_id);
    
    // Execute tool...
  }
};
```

#### Timeout Policies

**Recommended Timeouts:**
| Tool Type | Default Timeout | Rationale |
|-----------|-----------------|-----------|
| `ui.showModal` | 60s | User reads message + clicks button |
| `ui.requestApproval` | 5-15 min | Approver may be in meeting, need time to review |
| `ui.uploadFile` | 120s | User finds file + uploads (depends on file size) |
| `ui.confirmAction` | 30s | Quick yes/no decision |

**Timeout Handling:**
```typescript
async function executeToolWithTimeout(
  tool: string,
  args: any,
  timeout_ms: number
): Promise<any> {
  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Timeout: ${tool}`)), timeout_ms)
  );
  
  const toolPromise = executeToolHandler(tool, args);
  
  try {
    return await Promise.race([toolPromise, timeoutPromise]);
  } catch (error) {
    // Log timeout, optionally notify backend
    logToolTimeout({ tool, args, error });
    throw error;
  }
}
```

#### Distributed State Management

**Challenge:** Workflow state lives in backend, UI state in frontend. Reconciliation needed on reconnect.

**Solution:**
```typescript
// On WebSocket reconnect, fetch current execution state
ws.onopen = async () => {
  const activeExecutions = await fetch('/api/executions?status=running');
  const executions = await activeExecutions.json();
  
  // Restore UI state for active workflows
  executions.forEach((exec) => {
    restoreExecutionUI(exec.id, exec.current_step, exec.pending_tool_calls);
  });
};

// Backend persists tool call state
// (Backend implementation)
class ToolCallManager:
    async def send_tool_call(self, execution_id, tool, args):
        call_id = str(uuid.uuid4())
        
        # Persist to DB before sending (crash-safe)
        await db.tool_calls.insert({
            "call_id": call_id,
            "execution_id": execution_id,
            "tool": tool,
            "args": args,
            "status": "pending",
            "created_at": datetime.utcnow()
        })
        
        # Send to frontend
        await websocket.send_json({
            "type": "tool_call_requested",
            "call_id": call_id,
            "tool": tool,
            "args": args
        })
        
        # Wait for response (with timeout)
        return await self.wait_for_response(call_id, timeout=300)
```

#### Error Recovery

**Frontend Crash:** Backend should timeout tool calls and mark them as failed.
```python
# Backend timeout handler
async def handle_tool_call_timeout(call_id: str):
    await db.tool_calls.update(call_id, status="timeout")
    await fail_execution_step(call_id, error="Tool call timeout")
```

**Backend Crash:** Frontend should detect WebSocket disconnect and alert user.
```typescript
ws.onclose = () => {
  showReconnectingBanner();
  attemptReconnect();
};

async function attemptReconnect() {
  const backoff = [1000, 2000, 5000, 10000];  // Exponential backoff
  for (const delay of backoff) {
    await sleep(delay);
    try {
      await reconnectWebSocket();
      hideReconnectingBanner();
      return;
    } catch (error) {
      console.error('Reconnect failed, retrying...');
    }
  }
  showErrorModal('Lost connection to server. Please refresh.');
}
```

---

## Section 4: Decision Framework

### Choosing the Right Integration Mode

Use this decision tree to select the appropriate integration mode for your use case:

```
START: How does your frontend need to integrate with N3?
│
├─► Do you need HUMAN-IN-THE-LOOP interactions?
│   (approvals, file uploads, UI confirmations)
│   │
│   ├─► YES → Use HYBRID MODE
│   │         └─► Backend runs workflows, frontend provides UI tools
│   │
│   └─► NO → Continue ↓
│
├─► Do you need SUB-50ms LATENCY for workflows?
│   │
│   ├─► YES → Do you have LONG-RUNNING workflows (>30s)?
│   │         │
│   │         ├─► YES → Use HYBRID MODE (backend for long workflows)
│   │         │         + EMBEDDED MODE (for fast, synchronous features)
│   │         │
│   │         └─► NO → Use EMBEDDED MODE (Next.js API routes, Server Components)
│   │
│   └─► NO → Continue ↓
│
├─► Do you need MULTI-PLATFORM support?
│   (web, mobile, desktop, CLI)
│   │
│   ├─► YES → Use REMOTE SERVICE MODE
│   │         └─► Single backend serves all clients
│   │
│   └─► NO → Continue ↓
│
├─► Do you need to SCALE INDEPENDENTLY?
│   (frontend vs backend scaling, serverless edge)
│   │
│   ├─► YES → Use REMOTE SERVICE MODE
│   │         └─► Backend scales based on workflow load
│   │
│   └─► NO → Continue ↓
│
└─► DEFAULT → Use EMBEDDED MODE
              └─► Simplest for new Next.js/Remix apps
```

---

### Mode Selection Matrix

| Criteria | Remote Service | Embedded Runtime | Hybrid |
|----------|---------------|------------------|--------|
| **Latency target** | 100-500ms | 5-50ms | 100-500ms (backend) + variable (UI tools) |
| **Human-in-the-loop** | ❌ Not ideal | ❌ Not supported | ✅ Primary use case |
| **Multi-platform** | ✅ Ideal | ❌ Web-only (BFF tied to frontend) | ✅ Ideal |
| **Independent scaling** | ✅ Yes | ❌ No (frontend scales with backend) | ✅ Yes |
| **Long-running workflows** | ✅ Yes (background jobs) | ⚠️ Limited (serverless timeouts) | ✅ Yes |
| **Deployment complexity** | Medium (2 services) | Low (1 service) | High (WebSocket infra) |
| **Cold start impact** | Low (dedicated backend) | High (edge/serverless) | Low (backend) |
| **Debugging complexity** | Medium (2 services) | Low (single process) | High (distributed tracing) |
| **When to use** | Multi-platform, independent services | New Next.js apps, low-latency sync features | Approval workflows, UI gates |

---

### Migration Paths

#### Path 1: Remote Service → Embedded Runtime

**Scenario:** You started with a remote N3 backend, but want to reduce latency for specific features by moving them to the BFF.

**Strategy:**
1. **Identify latency-sensitive workflows** (e.g., search, autocomplete, form validation)
2. **Create a Next.js API route** that runs N3 in-process
3. **Migrate one workflow at a time** (feature flag to toggle between remote/embedded)
4. **Keep remote backend for long-running workflows** (reports, batch jobs)

**Example Migration:**
```typescript
// Before: Remote service
async function searchProducts(query: string) {
  const response = await fetch('https://api.example.com/workflows/search/execute', {
    method: 'POST',
    body: JSON.stringify({ input: { query } }),
  });
  return response.json();  // 200-300ms latency
}

// After: Embedded runtime
import { getN3Runtime } from '@/lib/n3';

async function searchProducts(query: string) {
  const runtime = getN3Runtime();
  const result = await runtime.executeWorkflow('search', {
    input: { query },
  });
  return result;  // 10-20ms latency
}
```

**Rollout Plan:**
```typescript
// Feature flag for gradual migration
const USE_EMBEDDED_SEARCH = process.env.FEATURE_EMBEDDED_SEARCH === 'true';

async function searchProducts(query: string) {
  if (USE_EMBEDDED_SEARCH) {
    return searchProductsEmbedded(query);
  } else {
    return searchProductsRemote(query);
  }
}
```

---

#### Path 2: Embedded Runtime → Hybrid Mode

**Scenario:** You started with embedded N3 in Next.js, but now need human approval workflows.

**Strategy:**
1. **Keep embedded mode for synchronous workflows** (dashboard, search, CRUD)
2. **Add WebSocket infrastructure** for approval workflows
3. **Register UI tools** (`ui.requestApproval`, `ui.uploadFile`)
4. **Run approval workflows remotely** while keeping fast workflows embedded

**Architecture Evolution:**
```
Before:
┌─────────────┐
│  Next.js    │
│  + N3       │  ← All workflows in-process
└─────────────┘

After (Hybrid):
┌─────────────┐         ┌──────────────┐
│  Next.js    │  WS     │  N3 Backend  │
│  + N3       │◄───────►│              │  ← Approval workflows
│  (fast)     │         │  (approvals) │
└─────────────┘         └──────────────┘
```

**Implementation:**
```typescript
// Workflow routing logic
async function executeWorkflow(workflowId: string, input: any) {
  const APPROVAL_WORKFLOWS = ['payment_approval', 'content_moderation'];
  
  if (APPROVAL_WORKFLOWS.includes(workflowId)) {
    // Execute remotely with UI tools
    return executeWorkflowRemote(workflowId, input);
  } else {
    // Execute in-process (fast)
    const runtime = getN3Runtime();
    return runtime.executeWorkflow(workflowId, { input });
  }
}
```

---

#### Path 3: Remote Service → Hybrid Mode

**Scenario:** You have a remote N3 backend, now need to add UI interactions.

**Strategy:**
1. **Keep remote backend as-is** (no changes to existing workflows)
2. **Add WebSocket endpoint** to backend
3. **Frontend registers UI tools** on connect
4. **Create new workflows that use UI tools** (start with one pilot workflow)

**Backend Changes:**
```python
# app/websocket.py (new file)
from fastapi import WebSocket
from namel3ss import Runtime

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Receive tool registration
    tools_msg = await websocket.receive_json()
    registered_tools = tools_msg['tools']
    
    # 2. Create runtime with UI tools
    runtime = Runtime()
    for tool in registered_tools:
        runtime.register_tool(
            name=tool['id'],
            handler=lambda args: send_tool_call_and_wait(websocket, tool['id'], args),
            schema=tool['schema']
        )
    
    # 3. Execute workflows that use UI tools
    while True:
        msg = await websocket.receive_json()
        if msg['type'] == 'execute_workflow':
            result = await runtime.execute_workflow(
                msg['workflow_id'],
                input=msg['input']
            )
            await websocket.send_json({'type': 'result', 'data': result})
```

**Frontend Changes:**
```typescript
// Add WebSocket connection (no changes to existing REST API)
const ws = new WebSocket('wss://api.example.com/ws');

ws.onopen = () => {
  // Register UI tools
  ws.send(JSON.stringify({
    type: 'register_tools',
    tools: [
      { id: 'ui.requestApproval', schema: {...} }
    ]
  }));
};

// Handle tool calls
ws.onmessage = async (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'tool_call_requested') {
    const result = await handleToolCall(msg.tool, msg.args);
    ws.send(JSON.stringify({
      type: 'tool_call_response',
      call_id: msg.call_id,
      result
    }));
  }
};
```

---

### Multi-Mode Architectures

Some teams use **multiple modes simultaneously** for different parts of their application:

#### Architecture Example: E-commerce Platform

```
┌─────────────────────────────────────────────────────────────────────┐
│                     E-commerce Platform Architecture                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Next.js Frontend                                            │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                              │  │
│  │  EMBEDDED MODE (in-process):                                │  │
│  │  • Product search (10ms latency)                            │  │
│  │  • Shopping cart validation                                 │  │
│  │  • Inventory checks                                         │  │
│  │                                                              │  │
│  │  REMOTE MODE (via REST):                                    │  │
│  │  • Order processing                                         │  │
│  │  • Inventory sync (background job)                          │  │
│  │  • Analytics workflows                                      │  │
│  │                                                              │  │
│  │  HYBRID MODE (via WebSocket):                               │  │
│  │  • Payment approval (>$10k orders)                          │  │
│  │  • Fraud review (UI tool: "ui.requestReview")              │  │
│  │  • Customer verification (UI tool: "ui.uploadID")          │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                │                    │                     │
│         │ (in-process)   │ (REST API)         │ (WebSocket)         │
│         ↓                ↓                    ↓                     │
│  ┌──────────┐     ┌──────────────────┐  ┌──────────────────┐      │
│  │  N3 in   │     │   N3 Backend     │  │  N3 Backend      │      │
│  │  Next.js │     │   (FastAPI)      │  │  (WebSocket)     │      │
│  └──────────┘     └──────────────────┘  └──────────────────┘      │
│                            │                     │                  │
│                            ↓                     ↓                  │
│                    ┌───────────────────────────────┐               │
│                    │      PostgreSQL Database      │               │
│                    └───────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Routing Logic:**
```typescript
// Centralized workflow executor
async function executeWorkflow(
  workflowId: string,
  input: any,
  options?: { mode?: 'embedded' | 'remote' | 'hybrid' }
) {
  // Determine mode (explicit or auto-detect)
  const mode = options?.mode ?? detectMode(workflowId);
  
  switch (mode) {
    case 'embedded':
      return executeEmbedded(workflowId, input);
    case 'remote':
      return executeRemote(workflowId, input);
    case 'hybrid':
      return executeHybrid(workflowId, input);
  }
}

// Auto-detect mode based on workflow characteristics
function detectMode(workflowId: string): 'embedded' | 'remote' | 'hybrid' {
  const EMBEDDED_WORKFLOWS = ['search', 'cart_validation', 'inventory_check'];
  const HYBRID_WORKFLOWS = ['payment_approval', 'fraud_review', 'customer_verification'];
  
  if (EMBEDDED_WORKFLOWS.includes(workflowId)) {
    return 'embedded';
  } else if (HYBRID_WORKFLOWS.includes(workflowId)) {
    return 'hybrid';
  } else {
    return 'remote';  // Default
  }
}
```

---

### Mode Selection Anti-Patterns

**Anti-Pattern 1: Using Embedded Mode for Long-Running Workflows**
```typescript
// ❌ BAD: 5-minute report generation in Next.js API route
export async function GET(request: NextRequest) {
  const runtime = getN3Runtime();
  const report = await runtime.executeWorkflow('generate_report', {
    input: { start_date: '2025-01-01', end_date: '2025-12-31' }
  });  // Times out after 60s (Vercel limit)
  return NextResponse.json(report);
}

// ✅ GOOD: Use remote backend for long-running workflows
export async function GET(request: NextRequest) {
  const response = await fetch('https://api.example.com/workflows/generate_report/execute', {
    method: 'POST',
    body: JSON.stringify({
      input: { start_date: '2025-01-01', end_date: '2025-12-31' }
    })
  });
  const { execution_id } = await response.json();
  
  // Poll for completion or use WebSocket for updates
  return NextResponse.json({ execution_id, status: 'processing' });
}
```

**Anti-Pattern 2: Using Hybrid Mode for Simple Workflows**
```typescript
// ❌ BAD: Hybrid mode for workflow that doesn't need UI tools
const ws = new WebSocket('wss://api.example.com/ws');
ws.send(JSON.stringify({
  type: 'execute_workflow',
  workflow_id: 'get_user_profile',  // Simple data fetch, no UI interaction
  input: { user_id: 123 }
}));

// ✅ GOOD: Use remote mode (REST API) for simple workflows
const response = await fetch('https://api.example.com/workflows/get_user_profile/execute', {
  method: 'POST',
  body: JSON.stringify({ input: { user_id: 123 } })
});
const result = await response.json();
```

**Anti-Pattern 3: Using Remote Mode for Ultra-Low-Latency Features**
```typescript
// ❌ BAD: Remote service for autocomplete (200ms latency)
async function autocompleteSearch(query: string) {
  const response = await fetch('https://api.example.com/workflows/autocomplete/execute', {
    method: 'POST',
    body: JSON.stringify({ input: { query } })
  });
  return response.json();  // Network round-trip adds 150-200ms
}

// ✅ GOOD: Use embedded mode for autocomplete (10-20ms)
async function autocompleteSearch(query: string) {
  const runtime = getN3Runtime();
  const result = await runtime.executeWorkflow('autocomplete', {
    input: { query }
  });
  return result;  // In-process, no network latency
}
```

---

## Section 5: Cross-Cutting Concerns

This section covers implementation details that apply across **all integration modes**: authentication, observability, error handling, testing, and security.

---

### Authentication & Authorization

#### JWT-Based Auth Flow (All Modes)

**Pattern:** Frontend obtains JWT, includes it in requests to N3 backend (or passes to embedded runtime context).

```typescript
// 1. Frontend: User logs in, receives JWT
const loginResponse = await fetch('https://auth.example.com/login', {
  method: 'POST',
  body: JSON.stringify({ email, password })
});
const { access_token, user } = await loginResponse.json();

// Store token (httpOnly cookie or secure storage)
localStorage.setItem('access_token', access_token);

// 2. Include token in N3 workflow requests
async function executeWorkflowAuthenticated(workflowId: string, input: any) {
  const token = localStorage.getItem('access_token');
  
  // Remote Mode
  const response = await fetch(`https://api.example.com/workflows/${workflowId}/execute`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ input })
  });
  
  return response.json();
}

// 3. Backend: Verify JWT and extract user context
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=["HS256"]
        )
        return payload  # {"user_id": 123, "tenant_id": "acme", "role": "admin"}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# 4. Pass user context to N3 workflow
@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteRequest,
    user: dict = Depends(verify_token)
):
    runtime = Runtime()
    result = await runtime.execute_workflow(
        workflow_id,
        input=request.input,
        context={
            "user_id": user["user_id"],
            "tenant_id": user["tenant_id"],
            "role": user["role"]
        }
    )
    return result
```

#### Embedded Mode Auth

**Pattern:** Use Next.js session in Server Components/API routes, pass to N3 context.

```typescript
// app/api/workflows/[id]/route.ts
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { getN3Runtime } from '@/lib/n3';

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  // 1. Verify session
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  
  // 2. Execute with user context
  const runtime = getN3Runtime();
  const { input } = await request.json();
  
  const result = await runtime.executeWorkflow(params.id, {
    input,
    context: {
      user_id: session.user.id,
      tenant_id: session.user.tenantId,
      email: session.user.email
    }
  });
  
  return NextResponse.json(result);
}
```

#### WebSocket Auth (Hybrid Mode)

**Pattern:** Authenticate WebSocket connection on initial handshake.

```python
# Backend: Verify token on WebSocket connect
from fastapi import WebSocket, WebSocketDisconnect, Query
import jwt

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)  # wss://api.example.com/ws?token=<jwt>
):
    # 1. Verify token before accepting connection
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = payload["user_id"]
        tenant_id = payload["tenant_id"]
    except jwt.InvalidTokenError:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    await websocket.accept()
    
    # 2. Store user context for this connection
    connection_context = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "websocket": websocket
    }
    
    # 3. All workflows on this connection use this context
    try:
        while True:
            msg = await websocket.receive_json()
            if msg["type"] == "execute_workflow":
                result = await runtime.execute_workflow(
                    msg["workflow_id"],
                    input=msg["input"],
                    context=connection_context
                )
                await websocket.send_json({"type": "result", "data": result})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user_id={user_id}")
```

```typescript
// Frontend: Include token in WebSocket URL
const token = localStorage.getItem('access_token');
const ws = new WebSocket(`wss://api.example.com/ws?token=${token}`);

ws.onerror = (event) => {
  if (event.code === 4001) {
    console.error('Authentication failed');
    redirectToLogin();
  }
};
```

---

### Multi-Tenancy

#### Row-Level Security (All Modes)

**Pattern:** Use tenant_id from auth context to filter data in N3 tools.

```python
# Tool implementation with tenant isolation
async def get_customers_tool(args: dict, context: dict) -> list:
    tenant_id = context["tenant_id"]
    
    # Enforce tenant isolation in query
    customers = await db.customers.find({
        "tenant_id": tenant_id,  # Critical: Always filter by tenant
        "status": args.get("status", "active")
    })
    
    return customers

# Register tool in N3 runtime
runtime.register_tool(
    name="customers.list",
    handler=get_customers_tool,
    schema={...}
)
```

#### Tenant-Specific Workflows

**Pattern:** Load workflows from tenant-specific storage.

```python
# Load workflow definition based on tenant
async def get_workflow(workflow_id: str, tenant_id: str):
    # Check tenant-specific workflow first
    custom_workflow = await db.workflows.find_one({
        "id": workflow_id,
        "tenant_id": tenant_id
    })
    
    if custom_workflow:
        return custom_workflow  # Tenant customization
    
    # Fall back to default workflow
    default_workflow = await db.workflows.find_one({
        "id": workflow_id,
        "tenant_id": None  # Global default
    })
    
    return default_workflow

# Usage in execution
@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteRequest,
    user: dict = Depends(verify_token)
):
    workflow_def = await get_workflow(workflow_id, user["tenant_id"])
    
    runtime = Runtime()
    result = await runtime.execute_workflow_from_definition(
        workflow_def,
        input=request.input,
        context={"tenant_id": user["tenant_id"]}
    )
    return result
```

---

### Observability

#### Distributed Tracing (All Modes)

**Pattern:** Propagate trace context across frontend, N3 backend, and tool calls.

```typescript
// Frontend: Start trace and propagate
import { trace, context } from '@opentelemetry/api';

async function executeWorkflowWithTracing(workflowId: string, input: any) {
  const tracer = trace.getTracer('frontend');
  
  return tracer.startActiveSpan('execute_workflow', async (span) => {
    span.setAttribute('workflow_id', workflowId);
    
    // Extract trace context
    const traceContext = {};
    trace.inject(context.active(), traceContext);
    
    try {
      const response = await fetch(`https://api.example.com/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: {
          'traceparent': traceContext['traceparent'],  // W3C Trace Context
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input })
      });
      
      const result = await response.json();
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
      
    } catch (error) {
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR });
      throw error;
    } finally {
      span.end();
    }
  });
}
```

```python
# Backend: Continue trace
from opentelemetry import trace
from opentelemetry.propagate import extract

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteRequest,
    traceparent: str = Header(None)
):
    # Extract parent trace context
    carrier = {"traceparent": traceparent} if traceparent else {}
    ctx = extract(carrier)
    
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("n3_workflow_execution", context=ctx) as span:
        span.set_attribute("workflow_id", workflow_id)
        span.set_attribute("tenant_id", user["tenant_id"])
        
        runtime = Runtime()
        result = await runtime.execute_workflow(
            workflow_id,
            input=request.input,
            context={"span": span}  # Pass span to tools for sub-spans
        )
        
        span.set_attribute("execution_duration_ms", result.duration)
        return result
```

#### Structured Logging

**Pattern:** Include trace_id, execution_id, tenant_id in all logs.

```python
# Backend logging
import structlog

logger = structlog.get_logger()

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteRequest,
    user: dict = Depends(verify_token),
    trace_id: str = Header(None, alias="x-trace-id")
):
    execution_id = str(uuid.uuid4())
    
    # Bind context to logger
    log = logger.bind(
        execution_id=execution_id,
        workflow_id=workflow_id,
        tenant_id=user["tenant_id"],
        user_id=user["user_id"],
        trace_id=trace_id
    )
    
    log.info("workflow_execution_started", input_keys=list(request.input.keys()))
    
    try:
        runtime = Runtime()
        result = await runtime.execute_workflow(
            workflow_id,
            input=request.input,
            context={"logger": log}
        )
        
        log.info(
            "workflow_execution_completed",
            duration_ms=result.duration,
            steps_executed=len(result.steps)
        )
        return result
        
    except Exception as error:
        log.error(
            "workflow_execution_failed",
            error=str(error),
            error_type=type(error).__name__
        )
        raise
```

#### Metrics (Prometheus)

**Pattern:** Instrument key workflow metrics.

```python
# Backend metrics
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
workflow_executions = Counter(
    'n3_workflow_executions_total',
    'Total workflow executions',
    ['workflow_id', 'status', 'tenant_id']
)

workflow_duration = Histogram(
    'n3_workflow_duration_seconds',
    'Workflow execution duration',
    ['workflow_id', 'tenant_id'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
)

active_executions = Gauge(
    'n3_active_executions',
    'Currently executing workflows',
    ['workflow_id']
)

# Instrument execution
@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecuteRequest, user: dict = Depends(verify_token)):
    active_executions.labels(workflow_id=workflow_id).inc()
    
    start_time = time.time()
    status = "success"
    
    try:
        result = await runtime.execute_workflow(workflow_id, input=request.input)
        return result
        
    except Exception as error:
        status = "error"
        raise
        
    finally:
        duration = time.time() - start_time
        
        workflow_executions.labels(
            workflow_id=workflow_id,
            status=status,
            tenant_id=user["tenant_id"]
        ).inc()
        
        workflow_duration.labels(
            workflow_id=workflow_id,
            tenant_id=user["tenant_id"]
        ).observe(duration)
        
        active_executions.labels(workflow_id=workflow_id).dec()
```

---

### Error Handling

#### Standardized Error Response (All Modes)

**Pattern:** Return consistent error structure across all endpoints.

```typescript
// Shared error type
interface N3Error {
  error: {
    code: string;           // Machine-readable error code
    message: string;        // Human-readable message
    details?: any;          // Additional context
    trace_id?: string;      // For support/debugging
    execution_id?: string;  // Failed execution ID
  };
}

// Frontend error handler
async function executeWorkflowSafe(workflowId: string, input: any): Promise<any> {
  try {
    const response = await fetch(`https://api.example.com/workflows/${workflowId}/execute`, {
      method: 'POST',
      body: JSON.stringify({ input })
    });
    
    if (!response.ok) {
      const error: N3Error = await response.json();
      throw new WorkflowError(error.error);
    }
    
    return response.json();
    
  } catch (error) {
    if (error instanceof WorkflowError) {
      // Handle known N3 errors
      switch (error.code) {
        case 'WORKFLOW_NOT_FOUND':
          showNotification('Workflow not found', 'error');
          break;
        case 'TOOL_TIMEOUT':
          showNotification('Operation timed out, please try again', 'warning');
          break;
        case 'UNAUTHORIZED':
          redirectToLogin();
          break;
        default:
          showErrorModal({
            title: 'Workflow Error',
            message: error.message,
            trace_id: error.trace_id
          });
      }
    } else {
      // Handle unexpected errors
      console.error('Unexpected error:', error);
      showErrorModal({
        title: 'Unexpected Error',
        message: 'Something went wrong. Please try again.'
      });
    }
    
    throw error;
  }
}
```

```python
# Backend error handler
from fastapi import Request, status
from fastapi.responses import JSONResponse

class WorkflowException(Exception):
    def __init__(self, code: str, message: str, details: dict = None, status_code: int = 400):
        self.code = code
        self.message = message
        self.details = details
        self.status_code = status_code

@app.exception_handler(WorkflowException)
async def workflow_exception_handler(request: Request, exc: WorkflowException):
    trace_id = request.headers.get("x-trace-id")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "trace_id": trace_id
            }
        }
    )

# Usage in workflow execution
@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: ExecuteRequest):
    workflow = await db.workflows.find_one({"id": workflow_id})
    if not workflow:
        raise WorkflowException(
            code="WORKFLOW_NOT_FOUND",
            message=f"Workflow '{workflow_id}' not found",
            status_code=404
        )
    
    try:
        result = await runtime.execute_workflow(workflow_id, input=request.input)
        return result
    except ToolTimeoutError as error:
        raise WorkflowException(
            code="TOOL_TIMEOUT",
            message="Tool execution timed out",
            details={"tool": error.tool_name, "timeout_ms": error.timeout},
            status_code=408
        )
```

---

### Testing Strategies

#### Unit Testing (Embedded Mode)

**Pattern:** Test N3 workflows in isolation with mocked tools.

```typescript
// tests/workflows/search.test.ts
import { Runtime } from '@namel3ss/sdk';
import { describe, it, expect, vi } from 'vitest';

describe('Search Workflow', () => {
  it('should execute search workflow with mocked database', async () => {
    const runtime = new Runtime();
    
    // Mock database tool
    const mockDbSearch = vi.fn().mockResolvedValue([
      { id: 1, name: 'Product A' },
      { id: 2, name: 'Product B' }
    ]);
    
    runtime.registerTool({
      name: 'db.search',
      handler: mockDbSearch,
      schema: { query: { type: 'string' } }
    });
    
    // Execute workflow
    const result = await runtime.executeWorkflow('search', {
      input: { query: 'product' }
    });
    
    // Assertions
    expect(mockDbSearch).toHaveBeenCalledWith(
      { query: 'product' },
      expect.any(Object)  // context
    );
    expect(result.output).toHaveLength(2);
    expect(result.output[0].name).toBe('Product A');
  });
  
  it('should handle search errors gracefully', async () => {
    const runtime = new Runtime();
    
    runtime.registerTool({
      name: 'db.search',
      handler: async () => {
        throw new Error('Database connection failed');
      }
    });
    
    await expect(
      runtime.executeWorkflow('search', { input: { query: 'test' } })
    ).rejects.toThrow('Database connection failed');
  });
});
```

#### Integration Testing (Remote Mode)

**Pattern:** Test full frontend → backend → N3 flow with real services.

```typescript
// tests/integration/workflow-execution.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

describe('Workflow Execution API', () => {
  let authToken: string;
  
  beforeAll(async () => {
    // Authenticate test user
    const loginResponse = await fetch('http://localhost:8000/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        email: 'test@example.com',
        password: 'test-password'
      })
    });
    const { access_token } = await loginResponse.json();
    authToken = access_token;
  });
  
  it('should execute workflow and return result', async () => {
    const response = await fetch('http://localhost:8000/workflows/get_user/execute', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: { user_id: 123 }
      })
    });
    
    expect(response.status).toBe(200);
    
    const result = await response.json();
    expect(result).toMatchObject({
      execution_id: expect.any(String),
      status: 'completed',
      output: {
        id: 123,
        name: expect.any(String),
        email: expect.any(String)
      },
      duration: expect.any(Number)
    });
  });
  
  it('should return 401 for unauthenticated requests', async () => {
    const response = await fetch('http://localhost:8000/workflows/get_user/execute', {
      method: 'POST',
      body: JSON.stringify({ input: { user_id: 123 } })
    });
    
    expect(response.status).toBe(401);
  });
  
  it('should enforce tenant isolation', async () => {
    // Execute workflow as tenant A
    const response = await fetch('http://localhost:8000/workflows/list_customers/execute', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ input: {} })
    });
    
    const result = await response.json();
    
    // Verify all results belong to tenant A
    result.output.forEach((customer: any) => {
      expect(customer.tenant_id).toBe('tenant-a');
    });
  });
});
```

#### E2E Testing (Hybrid Mode)

**Pattern:** Test WebSocket tool call flow with Playwright.

```typescript
// tests/e2e/approval-workflow.spec.ts
import { test, expect } from '@playwright/test';

test('approval workflow with UI gate', async ({ page, context }) => {
  // 1. Login
  await page.goto('http://localhost:3000/login');
  await page.fill('input[name="email"]', 'admin@example.com');
  await page.fill('input[name="password"]', 'password');
  await page.click('button[type="submit"]');
  
  // 2. Navigate to payment page
  await page.goto('http://localhost:3000/payments/new');
  await page.fill('input[name="amount"]', '15000');
  await page.fill('input[name="vendor"]', 'Acme Corp');
  
  // 3. Submit payment (triggers approval workflow)
  await page.click('button:has-text("Submit Payment")');
  
  // 4. Wait for approval modal (triggered by ui.requestApproval tool)
  const modal = page.locator('[data-testid="approval-modal"]');
  await expect(modal).toBeVisible({ timeout: 10000 });
  
  // 5. Verify modal content
  await expect(modal.locator('.amount')).toHaveText('$15,000.00');
  await expect(modal.locator('.vendor')).toHaveText('Acme Corp');
  
  // 6. Approve payment
  await modal.locator('button:has-text("Approve")').click();
  
  // 7. Verify workflow completed
  await expect(page.locator('.success-message')).toHaveText('Payment approved and processed');
  
  // 8. Verify payment appears in history
  await page.goto('http://localhost:3000/payments');
  const paymentRow = page.locator('tr', { hasText: 'Acme Corp' });
  await expect(paymentRow.locator('.status')).toHaveText('Completed');
});

test('approval workflow timeout', async ({ page }) => {
  await page.goto('http://localhost:3000/payments/new');
  await page.fill('input[name="amount"]', '15000');
  await page.click('button:has-text("Submit Payment")');
  
  // Wait for modal
  const modal = page.locator('[data-testid="approval-modal"]');
  await expect(modal).toBeVisible();
  
  // Don't approve (wait for timeout)
  await expect(page.locator('.error-message')).toHaveText(
    'Approval timeout - please try again',
    { timeout: 60000 }  // 1 minute timeout
  );
});
```

---

### Security Best Practices

#### Input Validation

**Pattern:** Validate all workflow inputs against schema before execution.

```python
# Backend input validation
from pydantic import BaseModel, Field, validator

class WorkflowInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    filters: dict = Field(default_factory=dict)
    
    @validator('query')
    def validate_query(cls, v):
        # Prevent SQL injection patterns
        forbidden = ['--', ';', 'DROP', 'DELETE', 'UPDATE']
        if any(pattern in v.upper() for pattern in forbidden):
            raise ValueError('Invalid query: contains forbidden patterns')
        return v

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: ExecuteRequest
):
    # Validate input against workflow schema
    workflow = await db.workflows.find_one({"id": workflow_id})
    
    try:
        validated_input = WorkflowInput(**request.input)
    except ValidationError as error:
        raise WorkflowException(
            code="INVALID_INPUT",
            message="Input validation failed",
            details=error.errors(),
            status_code=400
        )
    
    result = await runtime.execute_workflow(
        workflow_id,
        input=validated_input.dict()
    )
    return result
```

#### Rate Limiting

**Pattern:** Apply rate limits per tenant/user to prevent abuse.

```python
# Backend rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/workflows/{workflow_id}/execute")
@limiter.limit("100/minute")  # 100 requests per minute per IP
async def execute_workflow(
    request: Request,
    workflow_id: str,
    execute_request: ExecuteRequest,
    user: dict = Depends(verify_token)
):
    # Additional per-tenant rate limit
    tenant_key = f"tenant:{user['tenant_id']}"
    tenant_limit = await redis.get(tenant_key)
    
    if tenant_limit and int(tenant_limit) > 1000:  # 1000 executions per hour
        raise WorkflowException(
            code="RATE_LIMIT_EXCEEDED",
            message="Tenant rate limit exceeded",
            status_code=429
        )
    
    await redis.incr(tenant_key)
    await redis.expire(tenant_key, 3600)  # 1 hour TTL
    
    result = await runtime.execute_workflow(workflow_id, input=execute_request.input)
    return result
```

#### Secrets Management

**Pattern:** Never pass secrets in workflow input; inject from secure storage.

```python
# Backend secrets injection
import boto3

secrets_client = boto3.client('secretsmanager')

async def get_secret(secret_name: str) -> str:
    response = secrets_client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Tool that needs secrets
async def send_email_tool(args: dict, context: dict) -> dict:
    # Get secret from AWS Secrets Manager (not from input)
    smtp_password = await get_secret(f"smtp_password_{context['tenant_id']}")
    
    # Use secret
    await send_email(
        to=args['recipient'],
        subject=args['subject'],
        body=args['body'],
        smtp_password=smtp_password
    )
    
    return {"status": "sent"}

# Frontend NEVER sends secrets
async function sendEmailWorkflow(recipient: string, subject: string, body: string) {
  // ❌ WRONG: Don't send secrets in input
  // const input = { recipient, subject, body, smtp_password: 'secret123' };
  
  // ✅ CORRECT: Backend fetches secrets
  const input = { recipient, subject, body };
  
  return executeWorkflow('send_email', input);
}
```

---

