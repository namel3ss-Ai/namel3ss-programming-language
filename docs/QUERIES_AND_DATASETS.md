# Queries and Dataset Operations Guide

**Purpose**: Learn how to filter, sort, aggregate, and transform data using Namel3ss dataset operations. This guide shows you the correct patterns to use instead of traditional SQL-like `query { }` blocks.

---

## Key Concept: Dataset Chaining

Namel3ss uses **dataset operations** rather than query blocks. Instead of writing:

❌ **Not Supported**:
```namel3ss
query {
    from jobs in "FileProcessingJob"
    where status == "pending"
    order by created_at desc
}
```

✅ **Use Dataset Operations**:
```namel3ss
dataset "PendingJobs" from postgres "jobs":
    filter by: status == "pending"
    order by: created_at desc
```

---

## Basic Filtering

### Single Filter

Filter by a single condition:

```namel3ss
dataset "ActiveUsers" from postgres "mydb":
    schema:
        id: uuid
        name: text
        status: text
        created_at: timestamp
    filter by: status == "active"
```

### Multiple Filters (AND Logic)

Stack multiple `filter by` clauses - they combine with AND:

```namel3ss
dataset "RecentActiveUsers" from postgres "mydb":
    schema:
        id: uuid
        name: text
        status: text
        last_login: timestamp
    filter by: status == "active"
    filter by: last_login > now() - days(7)
```

### Complex Filter Expressions

```namel3ss
dataset "HighValueOrders" from postgres "shop":
    schema:
        id: uuid
        customer_id: uuid
        total: decimal
        status: text
        created_at: timestamp
    filter by: (total > 100 and status == "completed") or (total > 500 and status == "pending")
    filter by: created_at > now() - days(30)
```

---

## Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `status == "active"` |
| `!=` | Not equal | `status != "deleted"` |
| `>` | Greater than | `age > 18` |
| `>=` | Greater or equal | `score >= 70` |
| `<` | Less than | `price < 100` |
| `<=` | Less or equal | `quantity <= 50` |

### String Operations

```namel3ss
dataset "SearchResults" from postgres "products":
    schema:
        id: uuid
        name: text
        description: text
    filter by: name contains "laptop"
    filter by: description not_contains "refurbished"
```

### Null Checks

```namel3ss
dataset "UsersWithPhone" from postgres "users":
    schema:
        id: uuid
        name: text
        phone: text?
    filter by: phone != null

dataset "UsersWithoutBio" from postgres "users":
    schema:
        id: uuid
        name: text
        bio: text?
    filter by: bio == null
```

### List/Array Operations

```namel3ss
dataset "AdminUsers" from postgres "users":
    schema:
        id: uuid
        name: text
        role: text
    filter by: role in ["admin", "superadmin", "moderator"]

dataset "RegularUsers" from postgres "users":
    schema:
        id: uuid
        name: text
        role: text
    filter by: role not_in ["admin", "superadmin"]
```

---

## Sorting and Ordering

### Basic Ordering

```namel3ss
dataset "LatestPosts" from postgres "blog":
    schema:
        id: uuid
        title: text
        published_at: timestamp
    order by: published_at desc
```

### Multiple Sort Fields

```namel3ss
dataset "SortedProducts" from postgres "inventory":
    schema:
        id: uuid
        category: text
        name: text
        price: decimal
    order by: category asc, price desc, name asc
```

### Ascending vs Descending

```namel3ss
# Ascending (default)
order by: created_at asc
order by: created_at  # 'asc' is implicit

# Descending
order by: created_at desc
```

---

## Pagination and Limiting

### Limit Results

```namel3ss
dataset "TopProducts" from postgres "products":
    schema:
        id: uuid
        name: text
        sales_count: integer
    order by: sales_count desc
    limit: 10
```

### Offset for Pagination

```namel3ss
dataset "ProductsPage2" from postgres "products":
    schema:
        id: uuid
        name: text
        price: decimal
    order by: name asc
    limit: 20
    offset: 20  # Skip first 20 results
```

### Pagination Pattern

```namel3ss
# Page 1 (results 1-20)
dataset "Page1" from postgres "items":
    limit: 20
    offset: 0

# Page 2 (results 21-40)
dataset "Page2" from postgres "items":
    limit: 20
    offset: 20

# Page 3 (results 41-60)
dataset "Page3" from postgres "items":
    limit: 20
    offset: 40
```

---

## Grouping and Aggregation

### Group By Single Field

```namel3ss
dataset "OrdersByStatus" from postgres "orders":
    schema:
        status: text
        order_count: integer
    group by: status
```

### Group By Multiple Fields

```namel3ss
dataset "SalesByRegionAndMonth" from postgres "sales":
    schema:
        region: text
        month: text
        total_sales: decimal
    group by: region, month
```

### Aggregation Functions

```namel3ss
dataset "UserStatistics" from postgres "users":
    schema:
        role: text
        user_count: integer
        avg_age: float
        min_created: timestamp
        max_created: timestamp
    group by: role
    aggregate:
        user_count: count(id)
        avg_age: avg(age)
        min_created: min(created_at)
        max_created: max(created_at)
```

### Available Aggregation Functions

| Function | Description | Example |
|----------|-------------|---------|
| `count(field)` | Count rows | `count(id)` |
| `sum(field)` | Sum values | `sum(amount)` |
| `avg(field)` | Average | `avg(price)` |
| `min(field)` | Minimum | `min(created_at)` |
| `max(field)` | Maximum | `max(score)` |
| `first(field)` | First value | `first(name)` |
| `last(field)` | Last value | `last(status)` |

### Aggregation with Filters

```namel3ss
dataset "ActiveUserStats" from postgres "users":
    schema:
        status: text
        user_count: integer
        avg_age: float
    filter by: deleted_at == null
    filter by: created_at > now() - days(365)
    group by: status
    aggregate:
        user_count: count(id)
        avg_age: avg(age)
```

---

## Joining Datasets

### Inner Join

```namel3ss
dataset "OrdersWithCustomers" from postgres "orders":
    schema:
        order_id: uuid
        customer_name: text
        customer_email: text
        order_total: decimal
        order_date: timestamp
    join: customers on customer_id
```

### Left Join

```namel3ss
dataset "AllCustomersWithOrders" from postgres "customers":
    schema:
        customer_id: uuid
        customer_name: text
        order_count: integer
    left_join: orders on id == orders.customer_id
    group by: customer_id
    aggregate:
        order_count: count(orders.id)
```

### Multiple Joins

```namel3ss
dataset "OrderDetails" from postgres "orders":
    schema:
        order_id: uuid
        customer_name: text
        product_name: text
        quantity: integer
        total: decimal
    join: customers on customer_id
    join: order_items on id == order_items.order_id
    join: products on order_items.product_id
```

---

## Date and Time Filtering

### Relative Time Filters

```namel3ss
dataset "RecentActivity" from postgres "logs":
    schema:
        id: uuid
        action: text
        created_at: timestamp
    filter by: created_at > now() - hours(24)

dataset "LastWeekOrders" from postgres "orders":
    schema:
        id: uuid
        total: decimal
        created_at: timestamp
    filter by: created_at > now() - days(7)

dataset "LastMonthUsers" from postgres "users":
    schema:
        id: uuid
        name: text
        created_at: timestamp
    filter by: created_at > now() - months(1)
```

### Date Range Filters

```namel3ss
dataset "Q1Orders" from postgres "orders":
    schema:
        id: uuid
        total: decimal
        order_date: date
    filter by: order_date >= date("2024-01-01")
    filter by: order_date <= date("2024-03-31")

dataset "TodayActivity" from postgres "events":
    schema:
        id: uuid
        event_type: text
        created_at: timestamp
    filter by: date(created_at) == today()
```

### Time Functions

| Function | Description | Example |
|----------|-------------|---------|
| `now()` | Current timestamp | `created_at < now()` |
| `today()` | Current date | `date == today()` |
| `hours(n)` | N hours | `now() - hours(6)` |
| `days(n)` | N days | `now() - days(30)` |
| `weeks(n)` | N weeks | `now() - weeks(2)` |
| `months(n)` | N months | `now() - months(3)` |
| `years(n)` | N years | `now() - years(1)` |

---

## Common Patterns

### Pattern 1: Dashboard Statistics

```namel3ss
dataset "DashboardStats" from postgres "orders":
    schema:
        status: text
        count: integer
        total_revenue: decimal
        avg_order_value: decimal
    filter by: created_at > now() - days(30)
    group by: status
    aggregate:
        count: count(id)
        total_revenue: sum(total)
        avg_order_value: avg(total)

page "Dashboard" at "/":
    show card "Monthly Stats":
        for stat in DashboardStats:
            show text "{{stat.status}}: {{stat.count}} orders"
            show text "Revenue: ${{stat.total_revenue}}"
```

### Pattern 2: Search and Filter

```namel3ss
dataset "ProductSearch" from postgres "products":
    schema:
        id: uuid
        name: text
        category: text
        price: decimal
        in_stock: boolean
    filter by: name contains ctx.query.search
    filter by: category == ctx.query.category or ctx.query.category == "all"
    filter by: in_stock == true
    order by: name asc

page "SearchProducts" at "/products/search":
    show form "SearchForm":
        fields:
            search:
                type: "text"
                placeholder: "Search products..."
            category:
                type: "select"
                options: ["all", "electronics", "clothing", "home"]
        
        on submit:
            # Filter updates dataset automatically
            refresh data
    
    show table:
        data: ProductSearch
        columns:
            - name
            - category
            - price
```

### Pattern 3: Leaderboard / Top N

```namel3ss
dataset "TopSellers" from postgres "users":
    schema:
        user_id: uuid
        username: text
        total_sales: decimal
        order_count: integer
    join: orders on id == orders.seller_id
    group by: user_id, username
    aggregate:
        total_sales: sum(orders.total)
        order_count: count(orders.id)
    order by: total_sales desc
    limit: 10

page "Leaderboard" at "/top-sellers":
    show list:
        data: TopSellers
        for seller in data:
            show card:
                show text "#{index + 1} {{seller.username}}"
                show text "Sales: ${{seller.total_sales}}"
                show text "Orders: {{seller.order_count}}"
```

### Pattern 4: Recent Activity Feed

```namel3ss
dataset "ActivityFeed" from postgres "activities":
    schema:
        id: uuid
        user_name: text
        action: text
        target: text
        created_at: timestamp
    join: users on user_id == users.id
    filter by: created_at > now() - hours(24)
    order by: created_at desc
    limit: 50

page "Dashboard" at "/":
    show card "Recent Activity":
        for activity in ActivityFeed:
            show text "{{activity.user_name}} {{activity.action}} {{activity.target}}"
            show text:
                text: "{{format_timestamp(activity.created_at, 'relative')}}"
                style: "text-muted"
```

### Pattern 5: Paginated List

```namel3ss
dataset "AllProducts" from postgres "products":
    schema:
        id: uuid
        name: text
        price: decimal
        created_at: timestamp
    order by: created_at desc
    limit: 20
    offset: ctx.query.page * 20

page "Products" at "/products":
    show table:
        data: AllProducts
        columns:
            - name
            - price
            - created_at
        
        pagination:
            page_size: 20
            current_page: ctx.query.page or 0
```

---

## Advanced Filtering

### Conditional Filters Based on Context

```namel3ss
dataset "FilteredOrders" from postgres "orders":
    schema:
        id: uuid
        customer_id: uuid
        status: text
        total: decimal
        created_at: timestamp
    # Apply filter only if user has specific role
    filter by: customer_id == ctx.user.id or ctx.user.role == "admin"
    filter by: status != "deleted"
```

### Dynamic Filters from Form Input

```namel3ss
page "OrderManagement" at "/orders":
    show form "FilterForm":
        fields:
            status:
                type: "select"
                options: ["all", "pending", "completed", "cancelled"]
            min_amount:
                type: "number"
                placeholder: "Min amount"
            max_amount:
                type: "number"
                placeholder: "Max amount"
    
    dataset "FilteredOrders" from postgres "orders":
        schema:
            id: uuid
            customer_name: text
            status: text
            total: decimal
        filter by: status == form.status or form.status == "all"
        filter by: total >= form.min_amount or form.min_amount == null
        filter by: total <= form.max_amount or form.max_amount == null
        order by: created_at desc
    
    show table:
        data: FilteredOrders
```

---

## Performance Tips

### 1. Filter Early

✅ **Good**: Filter before joining
```namel3ss
dataset "RecentOrders" from postgres "orders":
    filter by: created_at > now() - days(7)
    join: customers on customer_id
```

❌ **Less Efficient**: Join before filtering
```namel3ss
dataset "RecentOrders" from postgres "orders":
    join: customers on customer_id
    filter by: created_at > now() - days(7)
```

### 2. Use Indexes

Define indexes on frequently filtered fields:

```namel3ss
dataset "Orders" from postgres "orders":
    schema:
        id: uuid
        customer_id: uuid
        status: text
        created_at: timestamp
    indexes:
        - customer_id
        - status
        - created_at
```

### 3. Limit Results

Always use `limit` for large datasets:

```namel3ss
dataset "LatestActivity" from postgres "logs":
    schema:
        id: uuid
        action: text
        created_at: timestamp
    order by: created_at desc
    limit: 100  # Prevent loading millions of rows
```

### 4. Aggregate in Database

✅ **Good**: Aggregate in dataset
```namel3ss
dataset "OrderStats" from postgres "orders":
    group by: status
    aggregate:
        count: count(id)
        total: sum(amount)
```

❌ **Less Efficient**: Fetch all and count in UI
```namel3ss
dataset "AllOrders" from postgres "orders":
    # Don't do this for counting!
```

---

## Error Handling

### Common Errors

**Error**: "'query' blocks are not supported"

```namel3ss
query {  # ❌ Wrong
    from orders in "Orders"
    where status == "completed"
}
```

**Solution**: Use dataset operations
```namel3ss
dataset "CompletedOrders" from postgres "orders":  # ✅ Correct
    filter by: status == "completed"
```

**Error**: "Ternary operators not supported"

```namel3ss
dataset "Orders" from postgres "orders":
    filter by: status == "active" ? true : false  # ❌ Wrong
```

**Solution**: Use direct boolean expression
```namel3ss
dataset "Orders" from postgres "orders":
    filter by: status == "active"  # ✅ Correct
```

---

## Related Documentation

- [DATA_MODELS_GUIDE.md](./DATA_MODELS_GUIDE.md) - Dataset vs Frame concepts
- [STANDARD_LIBRARY.md](./STANDARD_LIBRARY.md) - Built-in functions
- [API_REFERENCE.md](./API_REFERENCE.md) - Complete API documentation
- [EXPRESSIONS_REFERENCE.md](./EXPRESSIONS_REFERENCE.md) - Expression syntax

---

**Last Updated**: November 28, 2025
