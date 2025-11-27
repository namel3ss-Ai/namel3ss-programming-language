# AI Customer Support Console

## Overview

The **AI Customer Support Console** is a production-ready example application that demonstrates how to build an AI-native support agent workspace using Namel3ss. This application showcases the language's ability to seamlessly integrate AI assistance, tool calling, real-time conversation interfaces, and professional dashboard components into a single cohesive system.

**Target Audience**: Support teams that want to augment human agents with AI assistance while maintaining human oversight and control.

**Key Value**: Demonstrates that Namel3ss is ideal for building real AI products, not just prototypes—with full integration from DSL to production-ready React UI.

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Support Agent                            │
│  (Human in control, AI assists with suggestions/tools)      │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│  Dashboard   │ │   Ticket    │ │  Session   │
│   Page       │ │  Workspace  │ │   Detail   │
│              │ │    Page     │ │    Page    │
│ • KPI Cards  │ │ • Chat UI   │ │ • Audit    │
│ • Ticket     │ │ • AI Panel  │ │ • Logs     │
│   List       │ │ • Tools     │ │ • Metrics  │
│ • Charts     │ │ • Forms     │ │ • Timeline │
└──────────────┘ └─────────────┘ └────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │    Namel3ss Runtime            │
        │  • Agent orchestration         │
        │  • Tool execution              │
        │  • Data bindings               │
        │  • Memory & state              │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   PostgreSQL Database          │
        │  • tickets                     │
        │  • customers                   │
        │  • orders                      │
        │  • interactions                │
        └────────────────────────────────┘
```

### Data Model

The application uses **five primary datasets**, all loaded from a PostgreSQL database (no hardcoded demo data):

1. **`tickets`**: Support tickets with status, priority, category, and assignment
   - Fields: `id`, `subject`, `status`, `priority`, `customer_id`, `agent_id`, `created_at`, `updated_at`, `category`, `urgency`

2. **`customers`**: Customer profiles with segment and lifetime value
   - Fields: `id`, `name`, `email`, `segment`, `lifetime_value`, `created_at`

3. **`orders`**: Order records with items and shipping details
   - Fields: `id`, `customer_id`, `status`, `total_amount`, `items`, `shipping_address`, `created_at`

4. **`interactions`**: Conversation messages from customers, agents, and AI
   - Fields: `id`, `ticket_id`, `speaker`, `message`, `timestamp`, `tokens`, `is_ai`

5. **`metrics`**: Aggregated KPIs computed via SQL query
   - Fields: `open_tickets`, `high_priority`, `avg_response_time_mins`, `avg_csat`, `tickets_24h`

### AI Agent

**Agent**: `support_assistant`  
**Model**: GPT-4o  
**Role**: Assistant to human support agents

**Capabilities**:
- Summarize customer conversations
- Suggest appropriate responses
- Propose tool calls with arguments
- Identify urgency and categorize issues
- Explain reasoning for all suggestions

**Tools Available**:
1. **`lookup_order`**: Fetch order details by ID
2. **`issue_refund`**: Process refunds (requires confirmation)
3. **`update_shipping_address`**: Update delivery address
4. **`tag_ticket`**: Add categorization tags

The agent **never acts autonomously**—all risky actions (refunds, address changes) require explicit human confirmation via modal dialogs.

---

## Core UI Components

This application demonstrates all major Namel3ss UI component categories:

### 1. **Navigation (Chrome Components)**

- **Sidebar**: Multi-level navigation with icons (Dashboard, Tickets, Analytics, Settings)
- **Navbar**: Top bar with logo, title, action buttons, and user menu
- **Breadcrumbs**: Hierarchical navigation trail on every page
- **Command Palette**: Quick navigation and search (inherited from chrome components)

### 2. **AI Semantic Components**

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `chat_thread` | Display conversation between customer, agent, and AI | • Message grouping by speaker<br>• Timestamps and avatars<br>• Token count display<br>• Copy message functionality<br>• Auto-scroll to latest |
| `agent_panel` | Show AI agent status and metrics | • Agent name and status<br>• Token usage (prompt/completion)<br>• Estimated cost<br>• Response latency<br>• Model info<br>• Available tools |
| `tool_call_view` | Display tool invocations with details | • Expandable inputs/outputs<br>• Timing information<br>• Success/failure status<br>• Copy call details<br>• Filter by tool/status |
| `log_view` | Structured log viewing (Session Detail page) | • Timestamp filtering<br>• Log level badges<br>• Source identification<br>• Full-text search |

### 3. **Data Display Components**

- **`stat_summary`**: KPI cards showing open tickets, response time, CSAT, etc.
- **`data_table`**: Sortable, filterable, paginated ticket list with row click navigation
- **`data_chart`**: Line chart showing ticket volume trends
- **`timeline`**: Event timeline for session audit trail

### 4. **Forms**

Four production forms demonstrating all field types:

| Form | Purpose | Fields |
|------|---------|--------|
| **Issue Refund** | Process refund requests | • Order ID (pattern validation)<br>• Amount (currency format)<br>• Reason (select dropdown)<br>• Notes (textarea) |
| **Update Shipping** | Modify delivery address | • Order ID<br>• Street, City, State, ZIP<br>• Pattern validation for ZIP |
| **Manage Ticket** | Update ticket state | • Status (select)<br>• Priority (radio group)<br>• Tags (multiselect)<br>• Assign To (select) |
| **Close Reason** | Document ticket resolution | • Resolution summary (textarea)<br>• Customer satisfaction (radio group) |

All forms use proper validation (required fields, patterns, min/max length) and display success/error messages.

### 5. **Feedback Components**

- **Modals**: Confirmation dialogs for risky operations
  - `refund_confirm`: Review refund details before processing
  - `close_ticket`: Confirm ticket closure with resolution form
  
- **Toasts**: Notification messages for async operations
  - `refund_success`: Success notification with refund ID
  - `refund_error`: Persistent error toast with retry action
  - `address_updated`: Success notification for address changes

- **Alerts**: Inline contextual messages
  - VIP customer policy notices
  - Warning messages for high-value refunds

---

## Pages & Workflows

### Page 1: Dashboard (`/dashboard`)

**Purpose**: High-level overview of support operations

**Components**:
- `stat_summary`: Four KPI cards (Open Tickets, High Priority, Avg Response Time, CSAT)
- `data_table`: Recent tickets with sortable/filterable columns
- `data_chart`: 7-day ticket volume trend

**Workflow**:
1. Agent opens dashboard to see current workload
2. Reviews KPIs for team performance
3. Sorts/filters ticket table to find specific issues
4. Clicks a row to navigate to ticket workspace

### Page 2: Ticket Workspace (`/ticket/:id`)

**Purpose**: Main interface for resolving individual tickets

**Layout**: Two-column grid
- **Left**: Conversation and tool activity
- **Right**: Agent panel and action forms

**Components**:
- `chat_thread`: Full conversation history (customer ↔ agent ↔ AI)
- `tool_call_view`: All tool invocations for this ticket
- `agent_panel`: AI assistant status and metrics
- Forms: Issue Refund, Update Shipping, Manage Ticket
- Modals: Refund confirmation, Close ticket
- Toasts: Success/error notifications
- Alert: VIP customer notice

**Workflow: Refund Request**:
1. Agent reads conversation in `chat_thread`
2. Customer requests refund for order #12345
3. AI suggests calling `lookup_order` tool → agent approves
4. Tool runs, displays order details in `tool_call_view`
5. Agent fills "Issue Refund" form (order ID, amount, reason)
6. Clicks "Preview Refund" → triggers `refund_confirm` modal
7. Modal shows review details: Order ID, Amount, Reason
8. Agent clicks "Confirm Refund" → modal action calls backend
9. Success: `refund_success` toast displays with refund ID
10. Tool call logged in `tool_call_view` with timing/status

**Workflow: Address Update**:
1. Customer reports wrong delivery address
2. Agent opens "Update Shipping" form
3. Enters order ID and new address (with ZIP validation)
4. Submits form → backend updates order
5. Success: `address_updated` toast confirms change
6. Tool call appears in `tool_call_view`

### Page 3: Session Detail (`/session/:session_id`)

**Purpose**: Audit trail and debugging for AI sessions

**Components**:
- Tabs layout with three views:
  - **Tool Calls**: `tool_call_view` with raw payload inspection
  - **Logs**: `log_view` with filtering and search
  - **Metrics**: `stat_summary` showing tokens, cost, duration, tool count
  - **Timeline**: `timeline` widget showing event sequence

**Use Case**: Compliance, debugging, or performance analysis

---

## How to Run

### Prerequisites

- Namel3ss CLI installed
- PostgreSQL database with sample data
- OpenAI API key for GPT-4o

### Setup

1. **Clone and navigate**:
   ```bash
   cd namel3ss-programming-language/examples
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export DATABASE_URL="postgresql://user:pass@localhost:5432/support_db"
   ```

3. **Initialize database** (sample schema):
   ```sql
   CREATE TABLE tickets (
     id SERIAL PRIMARY KEY,
     subject TEXT NOT NULL,
     status TEXT DEFAULT 'open',
     priority TEXT DEFAULT 'medium',
     customer_id INTEGER,
     agent_id INTEGER,
     created_at TIMESTAMP DEFAULT NOW(),
     updated_at TIMESTAMP DEFAULT NOW(),
     category TEXT,
     urgency TEXT
   );

   CREATE TABLE customers (
     id SERIAL PRIMARY KEY,
     name TEXT NOT NULL,
     email TEXT UNIQUE,
     segment TEXT,
     lifetime_value DECIMAL(10,2),
     created_at TIMESTAMP DEFAULT NOW()
   );

   CREATE TABLE orders (
     id SERIAL PRIMARY KEY,
     customer_id INTEGER REFERENCES customers(id),
     status TEXT DEFAULT 'pending',
     total_amount DECIMAL(10,2),
     items JSONB,
     shipping_address JSONB,
     created_at TIMESTAMP DEFAULT NOW()
   );

   CREATE TABLE interactions (
     id SERIAL PRIMARY KEY,
     ticket_id INTEGER REFERENCES tickets(id),
     speaker TEXT NOT NULL,
     message TEXT NOT NULL,
     timestamp TIMESTAMP DEFAULT NOW(),
     tokens INTEGER,
     is_ai BOOLEAN DEFAULT FALSE
   );

   -- Insert sample data...
   INSERT INTO customers (name, email, segment, lifetime_value) VALUES
     ('Alice Johnson', 'alice@example.com', 'VIP', 15000),
     ('Bob Smith', 'bob@example.com', 'Standard', 2500);

   INSERT INTO tickets (subject, status, priority, customer_id, category, urgency) VALUES
     ('Refund request for order #12345', 'open', 'high', 1, 'billing', 'high'),
     ('Product not working as described', 'open', 'medium', 2, 'technical', 'medium');
   ```

### Build and Run

```bash
# Generate backend and frontend
namel3ss generate examples/ai-customer-support-console.ai support_console_output

# Start backend (FastAPI)
cd support_console_output/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# In another terminal, start frontend (React + Vite)
cd support_console_output/frontend
npm install
npm run dev  # Opens at http://localhost:5173
```

### Testing

Access the application:
- **Dashboard**: http://localhost:5173/dashboard
- **Ticket Workspace**: http://localhost:5173/ticket/1
- **Session Detail**: http://localhost:5173/session/abc123

---

## What This Example Demonstrates

### 1. **Complete Namel3ss Pipeline**

- ✅ **Parser**: Handles all component syntaxes (`chat_thread`, `tool_call_view`, `agent_panel`, forms, modals, etc.)
- ✅ **AST**: Properly structured nodes for AI components, chrome, data display, and feedback
- ✅ **IR**: Runtime-agnostic intermediate representation with design tokens
- ✅ **Codegen (Backend)**: FastAPI endpoints for pages, forms, tools, and datasets
- ✅ **Codegen (Frontend)**: React components for all UI elements with proper type safety
- ✅ **Runtime**: Data bindings, tool execution, agent orchestration

### 2. **AI-Native Features**

- **Agent Definition**: `support_assistant` with model, instructions, tools, and configuration
- **Tool Calling**: Four production tools with typed parameters and return values
- **Conversation UI**: Real-time `chat_thread` with message grouping and streaming support
- **Tool Visibility**: `tool_call_view` shows all invocations with inputs/outputs/timing
- **Agent Monitoring**: `agent_panel` tracks tokens, cost, latency, and status

### 3. **Professional UI Patterns**

- **Navigation**: Sidebar, navbar, breadcrumbs, command palette
- **Data Display**: Tables, charts, KPI cards, timelines
- **Forms**: Structured inputs with validation and error handling
- **Feedback**: Modals for confirmations, toasts for notifications, alerts for warnings
- **Layout**: Grid, stack, tabs for organizing complex interfaces

### 4. **Real-World Workflows**

- **Refund Flow**: Form → Modal confirmation → Tool call → Success toast → Audit log
- **Address Update**: Form → Validation → Backend update → Success notification
- **Ticket Management**: Status changes, priority updates, tagging, assignment
- **Session Audit**: Full transparency into AI decisions, tool calls, and logs

### 5. **Production-Ready Practices**

- ✅ **No Hardcoded Data**: All data from PostgreSQL datasets
- ✅ **Realistic Schemas**: tickets, customers, orders, interactions
- ✅ **Proper Validation**: Form field patterns, required checks, min/max lengths
- ✅ **Error Handling**: Success/error messages, retry actions, persistent error toasts
- ✅ **Human Oversight**: Confirmations for risky operations (refunds, closures)
- ✅ **Audit Trail**: Session detail page with logs, metrics, timeline

---

## Extension Ideas

This example can be extended to demonstrate additional Namel3ss features:

1. **Multi-Channel Support**:
   - Add `chat_thread` for email, chat, phone transcripts
   - Different `agent_panel` configurations per channel

2. **SLA Tracking**:
   - Add `stat_summary` for response SLA compliance
   - `alert` components for breached SLAs

3. **Advanced Agent Features**:
   - Multiple agents with handoffs
   - Memory systems (conversation, session, global)
   - Chain workflows for complex ticket resolution

4. **More Tools**:
   - `cancel_order`, `issue_store_credit`, `escalate_to_tier2`
   - External API calls (CRM, payment processor)

5. **Enhanced Observability**:
   - `log_view` with structured logging
   - `evaluation_result` for agent performance metrics
   - `diff_view` for before/after state changes

6. **Security & Compliance**:
   - Role-based access control for tools
   - Audit logs for all actions
   - PII masking in chat displays

---

## Component Reference

### Syntax Examples

**Chat Thread**:
```namel3ss
chat_thread "conversation":
  messages_binding: "ticket.interactions"
  group_by: "speaker"
  show_timestamps: true
  show_avatar: true
  auto_scroll: true
  enable_copy: true
  max_height: "500px"
```

**Tool Call View**:
```namel3ss
tool_call_view "tool_calls":
  calls_binding: "ticket.tool_calls"
  show_inputs: true
  show_outputs: true
  show_timing: true
  show_status: true
  variant: "list"
  expandable: true
```

**Agent Panel**:
```namel3ss
agent_panel "ai_agent":
  agent_binding: "support_assistant"
  metrics_binding: "assistant_metrics"
  show_status: true
  show_tokens: true
  show_cost: true
  show_model: true
  show_tools: true
```

**Stat Summary**:
```namel3ss
stat_summary "metrics" data_binding "dashboard_metrics":
  metrics:
    - label: "Open Tickets"
      value_binding: "open_tickets"
      format: number
      trend: "up"
      variant: "default"
```

**Modal with Form**:
```namel3ss
modal "refund_confirm":
  title: "Confirm Refund"
  size: md
  dismissible: true
  trigger: "show_refund_confirm_modal"
  content:
    show text "Review details before processing"
  actions:
    action "Cancel" variant "ghost"
    action "Confirm" variant "destructive" action "process_refund"
```

---

## Testing

See test files added for this example:

- **Parser Tests**: `tests/test_ai_support_console_parsing.py`
- **IR Tests**: `tests/test_ai_support_console_ir.py`
- **Codegen Tests**: `tests/test_ai_support_console_codegen.py`

Run tests:
```bash
pytest tests/test_ai_support_console*.py -v
```

---

## Summary

The **AI Customer Support Console** is a flagship example demonstrating that Namel3ss is production-ready for real AI applications. It showcases:

- ✅ Complete integration from DSL to React UI
- ✅ All major component categories (AI, chrome, data, forms, feedback)
- ✅ Realistic data flows (no hardcoded demo data)
- ✅ Production patterns (validation, error handling, confirmations)
- ✅ Human-in-the-loop AI (agent assists, human decides)
- ✅ Full observability (logs, metrics, audit trails)

This is not a toy example—it's a template for building real AI-powered support systems with Namel3ss.

---

**Related Documentation**:
- [AI Semantic Components](../docs/AI_COMPONENTS.md)
- [Chrome Components Guide](../docs/CHROME_COMPONENTS_GUIDE.md)
- [Data Display Components](../docs/DATA_DISPLAY_COMPONENTS.md)
- [Forms Reference](../docs/FORMS_REFERENCE.md)
- [Feedback Components Guide](../docs/FEEDBACK_COMPONENTS_GUIDE.md)
