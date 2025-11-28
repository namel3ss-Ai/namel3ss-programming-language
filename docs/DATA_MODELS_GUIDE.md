# Data Models Guide: Datasets vs Frames

**Purpose**: This guide clarifies when to use `dataset` vs `frame` for data modeling in Namel3ss, helping you choose the right construct for your use case.

---

## Quick Decision Guide

**Use `dataset` when:**
- Working with database-backed data (Postgres, MySQL, etc.)
- Need CRUD operations (create, read, update, delete)
- Want to filter, sort, and transform data with operations
- Building applications with persistent data storage

**Use `frame` when:**
- Defining analytical schemas for type checking
- Working with in-memory data transformations
- Need relationship definitions between entities
- Building data pipelines with strict typing

---

## Datasets: Database-Backed Data Models

### What is a Dataset?

A `dataset` represents data stored in an external database. It provides a declarative way to define:
- Connection to data sources (Postgres, MySQL, SQLite)
- Schema structure with field types
- Query operations (filter, group, order, aggregate)
- Relationships between datasets

### Basic Dataset Definition

```namel3ss
dataset "Users" from postgres "mydb":
    schema:
        id: uuid
        name: text
        email: text
        created_at: timestamp
        role: text
```

### Dataset Operations

#### Filtering Data

```namel3ss
dataset "ActiveUsers" from postgres "mydb":
    schema:
        id: uuid
        name: text
        status: text
        last_login: timestamp
    filter by: status == "active"
    filter by: last_login > now() - days(30)
```

#### Sorting and Ordering

```namel3ss
dataset "RecentUsers" from postgres "mydb":
    schema:
        id: uuid
        name: text
        created_at: timestamp
    order by: created_at desc
    limit: 50
```

#### Grouping and Aggregation

```namel3ss
dataset "UsersByRole" from postgres "mydb":
    schema:
        role: text
        user_count: integer
        avg_age: float
    group by: role
    aggregate:
        user_count: count(id)
        avg_age: avg(age)
```

#### Joining Datasets

```namel3ss
dataset "OrdersWithCustomers" from postgres "mydb":
    schema:
        order_id: uuid
        customer_name: text
        order_total: decimal
        order_date: timestamp
    join: customers on customer_id
```

### Dataset in Pages (Display Data)

```namel3ss
dataset "Products" from postgres "inventory":
    schema:
        id: uuid
        name: text
        price: decimal
        category: text
    filter by: active == true
    order by: name asc

page "ProductList" at "/products":
    show table:
        data: Products
        columns:
            - name
            - price
            - category
```

### Dataset Updates (Write Operations)

```namel3ss
page "CreateUser" at "/users/new":
    show form "UserForm":
        fields:
            name:
                type: "text"
                required: true
            email:
                type: "email"
                required: true
            role:
                type: "select"
                options: ["admin", "user", "viewer"]
        
        on submit:
            insert into Users:
                name: form.name
                email: form.email
                role: form.role
            
            go to page "UserList"
            show toast "User created successfully"
```

---

## Frames: Analytical Schemas

### What is a Frame?

A `frame` defines a structured schema for type-safe data analysis. It's primarily used for:
- Type checking in expressions
- Defining relationships between entities
- In-memory data transformations
- Analytics and reporting pipelines

### Basic Frame Definition

```namel3ss
frame "Customer":
    columns: id, name, email, created_at
    description: "Customer entity with contact information"
```

### Frame with Detailed Schema

```namel3ss
frame "Order":
    schema:
        order_id: uuid
        customer_id: uuid
        total_amount: decimal
        order_date: timestamp
        status: text
    
    indexes:
        - customer_id
        - order_date
    
    relationships:
        customer: foreign_key(customer_id) references Customer(id)
```

### Frame Constraints

```namel3ss
frame "Transaction":
    schema:
        id: uuid
        amount: decimal
        transaction_type: text
        created_at: timestamp
    
    constraints:
        - amount > 0
        - transaction_type in ["debit", "credit", "refund"]
```

### Frame vs Dataset: Key Differences

| Feature | Dataset | Frame |
|---------|---------|-------|
| **Data Source** | External database | Schema definition only |
| **Operations** | Filter, sort, aggregate, join | Type checking, validation |
| **Use Case** | CRUD apps, data storage | Analytics, type safety |
| **Runtime** | Queries database | In-memory validation |
| **Relationships** | Via joins | Via foreign keys |

---

## Common Patterns

### Pattern 1: Dataset for Storage, Frame for Validation

```namel3ss
# Frame defines the schema contract
frame "FileProcessingJob":
    schema:
        id: uuid
        filename: text
        status: text
        created_at: timestamp
        completed_at: timestamp?
        error_message: text?
    
    constraints:
        - status in ["pending", "running", "completed", "failed"]
        - filename != ""

# Dataset implements the storage
dataset "Jobs" from postgres "processing":
    schema:
        id: uuid
        filename: text
        status: text
        created_at: timestamp
        completed_at: timestamp
        error_message: text
```

### Pattern 2: Dataset Chaining for Complex Queries

Instead of `query { }` blocks (not supported), chain dataset operations:

```namel3ss
dataset "PendingJobs" from postgres "processing":
    schema:
        id: uuid
        filename: text
        status: text
        priority: integer
        created_at: timestamp
    filter by: status == "pending"
    filter by: created_at > now() - hours(24)
    order by: priority desc, created_at asc
    limit: 100
```

### Pattern 3: Aggregated Dashboard Data

```namel3ss
dataset "JobStatistics" from postgres "processing":
    schema:
        status: text
        count: integer
        avg_duration: float
        latest_job: timestamp
    group by: status
    aggregate:
        count: count(id)
        avg_duration: avg(completed_at - created_at)
        latest_job: max(created_at)

page "Dashboard" at "/":
    show card "Job Status":
        for stat in JobStatistics:
            show text "{{stat.status}}: {{stat.count}} jobs"
            show text "Average: {{format_duration(stat.avg_duration)}}"
```

---

## Best Practices

### 1. Choose the Right Construct

✅ **Good**: Use dataset for persistent data
```namel3ss
dataset "Orders" from postgres "shop":
    schema:
        id: uuid
        customer_id: uuid
        total: decimal
```

❌ **Bad**: Don't use `type` keyword (not supported)
```namel3ss
type "Order" {  # ❌ Error: 'type' keyword not supported
    id: uuid
    total: decimal
}
```

### 2. Define Clear Schemas

✅ **Good**: Explicit field types
```namel3ss
dataset "Products" from postgres "inventory":
    schema:
        id: uuid
        name: text
        price: decimal
        stock: integer
        active: boolean
```

❌ **Bad**: Missing schema
```namel3ss
dataset "Products" from postgres "inventory":
    # ❌ No schema - unclear what fields exist
```

### 3. Use Dataset Operations for Queries

✅ **Good**: Dataset chaining
```namel3ss
dataset "ActiveProducts" from postgres "inventory":
    schema:
        id: uuid
        name: text
        price: decimal
    filter by: active == true
    filter by: stock > 0
    order by: name asc
```

❌ **Bad**: Query blocks (not supported)
```namel3ss
query {  # ❌ Error: 'query' blocks not supported
    from products in "Products"
    where active == true
}
```

### 4. Leverage Frames for Type Safety

✅ **Good**: Frame with constraints
```namel3ss
frame "Invoice":
    schema:
        invoice_number: text
        amount: decimal
        due_date: timestamp
    constraints:
        - amount > 0
        - invoice_number != ""
```

---

## Field Types Reference

### Supported Types in Datasets

| Type | Description | Example |
|------|-------------|---------|
| `uuid` | Unique identifier | `01234567-89ab-cdef-0123-456789abcdef` |
| `text` | String values | `"John Doe"` |
| `integer` | Whole numbers | `42` |
| `decimal` | Decimal numbers | `19.99` |
| `float` | Floating-point | `3.14159` |
| `boolean` | True/false | `true`, `false` |
| `timestamp` | Date and time | `2024-03-15T10:30:00Z` |
| `date` | Date only | `2024-03-15` |
| `time` | Time only | `10:30:00` |
| `json` | JSON objects | `{"key": "value"}` |

### Optional Fields

Use `?` suffix for optional/nullable fields:

```namel3ss
dataset "Users" from postgres "mydb":
    schema:
        id: uuid
        name: text
        email: text
        phone: text?          # Optional
        bio: text?            # Optional
        deleted_at: timestamp?  # Optional
```

---

## Error Handling

### Common Errors

**Error**: "'type' keyword is not supported"
```namel3ss
type "User" {  # ❌ Wrong
    name: text
}
```

**Solution**: Use dataset or frame
```namel3ss
dataset "Users" from postgres "mydb":  # ✅ Correct
    schema:
        name: text
```

**Error**: "query blocks are not supported"
```namel3ss
query {  # ❌ Wrong
    from users in "Users"
}
```

**Solution**: Use dataset operations
```namel3ss
dataset "ActiveUsers" from postgres "mydb":  # ✅ Correct
    filter by: status == "active"
```

---

## Examples

### Complete CRUD Example

```namel3ss
# Define the dataset
dataset "Tasks" from postgres "app":
    schema:
        id: uuid
        title: text
        description: text
        status: text
        priority: integer
        created_at: timestamp
        updated_at: timestamp

# List page
page "TaskList" at "/tasks":
    show table:
        data: Tasks
        columns:
            - title
            - status
            - priority
            - created_at
    
    show button "New Task":
        on click:
            go to page "CreateTask"

# Create page
page "CreateTask" at "/tasks/new":
    show form "TaskForm":
        fields:
            title:
                type: "text"
                required: true
            description:
                type: "textarea"
            status:
                type: "select"
                options: ["todo", "in_progress", "done"]
            priority:
                type: "number"
                min: 1
                max: 5
        
        on submit:
            insert into Tasks:
                title: form.title
                description: form.description
                status: form.status
                priority: form.priority
                created_at: now()
                updated_at: now()
            
            go to page "TaskList"
            show toast "Task created!"

# Update page
page "EditTask" at "/tasks/{id}":
    show form "EditTaskForm":
        initial_data: Tasks.filter(id == ctx.route.params.id)
        
        fields:
            title:
                type: "text"
                required: true
            status:
                type: "select"
                options: ["todo", "in_progress", "done"]
            priority:
                type: "number"
        
        on submit:
            update Tasks:
                where: id == ctx.route.params.id
                set:
                    title: form.title
                    status: form.status
                    priority: form.priority
                    updated_at: now()
            
            go to page "TaskList"
```

---

## Related Documentation

- [QUERIES_AND_DATASETS.md](./QUERIES_AND_DATASETS.md) - Dataset operations and filtering
- [STANDARD_LIBRARY.md](./STANDARD_LIBRARY.md) - Built-in functions like `now()`
- [EXTENSIONS_GUIDE.md](./EXTENSIONS_GUIDE.md) - Custom data processing tools
- [API_REFERENCE.md](./API_REFERENCE.md) - Complete API documentation

---

**Last Updated**: November 28, 2025
