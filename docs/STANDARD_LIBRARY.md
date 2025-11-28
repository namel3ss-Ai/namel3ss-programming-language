# Standard Library: Built-in Functions

**Purpose**: Reference guide for all built-in functions available in Namel3ss expressions, filters, and templates.

---

## Date and Time Functions

### `now()`

Returns the current timestamp.

**Syntax**: `now()`

**Returns**: `timestamp`

**Examples**:
```namel3ss
dataset "RecentLogs" from postgres "logs":
    filter by: created_at > now() - hours(24)

page "Dashboard" at "/":
    show text "Last updated: {{now()}}"
```

### `today()`

Returns the current date (without time component).

**Syntax**: `today()`

**Returns**: `date`

**Examples**:
```namel3ss
dataset "TodaysOrders" from postgres "orders":
    filter by: date(created_at) == today()
```

### `format_timestamp(timestamp, format)`

Formats a timestamp according to the specified format.

**Syntax**: `format_timestamp(ts, format)`

**Parameters**:
- `ts` (timestamp): The timestamp to format
- `format` (string): Format string or preset

**Format Presets**:
- `"short"` → `"3/15/24, 2:30 PM"`
- `"medium"` → `"Mar 15, 2024, 2:30:45 PM"`
- `"long"` → `"March 15, 2024 at 2:30:45 PM EST"`
- `"full"` → `"Monday, March 15, 2024 at 2:30:45 PM Eastern Standard Time"`
- `"relative"` → `"2 hours ago"`, `"in 3 days"`
- `"iso"` → `"2024-03-15T14:30:45.000Z"`

**Custom Format Tokens**:
- `YYYY` - Full year (2024)
- `MM` - Month (01-12)
- `DD` - Day (01-31)
- `HH` - Hour 24h (00-23)
- `mm` - Minute (00-59)
- `ss` - Second (00-59)

**Examples**:
```namel3ss
page "OrderDetails" at "/orders/{id}":
    show text "Ordered: {{format_timestamp(order.created_at, 'medium')}}"
    show text "Due: {{format_timestamp(order.due_date, 'relative')}}"
    show text "Exact time: {{format_timestamp(order.created_at, 'YYYY-MM-DD HH:mm:ss')}}"
```

### `format_date(date, format)`

Formats a date value.

**Syntax**: `format_date(date, format)`

**Parameters**:
- `date` (date): The date to format
- `format` (string): Format string

**Examples**:
```namel3ss
show text "Birthday: {{format_date(user.birthday, 'MM/DD/YYYY')}}"
show text "Start date: {{format_date(project.start_date, 'long')}}"
```

### `format_duration(duration)`

Formats a duration (difference between two timestamps) in human-readable format.

**Syntax**: `format_duration(duration)`

**Parameters**:
- `duration` (number): Duration in seconds

**Returns**: Human-readable string like "2 hours 30 minutes", "5 days"

**Examples**:
```namel3ss
dataset "JobStats" from postgres "jobs":
    schema:
        id: uuid
        avg_duration: float
    aggregate:
        avg_duration: avg(completed_at - created_at)

page "Stats" at "/stats":
    for job in JobStats:
        show text "Average: {{format_duration(job.avg_duration)}}"
```

### Time Duration Helpers

#### `hours(n)`, `days(n)`, `weeks(n)`, `months(n)`, `years(n)`

Create time durations for relative time calculations.

**Syntax**: 
- `hours(n)` - N hours
- `days(n)` - N days  
- `weeks(n)` - N weeks
- `months(n)` - N months
- `years(n)` - N years

**Examples**:
```namel3ss
filter by: created_at > now() - hours(6)
filter by: expires_at < now() + days(7)
filter by: membership_start > now() - months(12)
```

---

## String Functions

### `format(value, pattern)`

Formats a value according to a pattern.

**Syntax**: `format(value, pattern)`

**Parameters**:
- `value` (any): Value to format
- `pattern` (string): Format pattern

**Examples**:
```namel3ss
show text "Price: {{format(product.price, '$0,0.00')}}"
show text "Percentage: {{format(completion_rate, '0.0%')}}"
show text "Phone: {{format(user.phone, '(000) 000-0000')}}"
```

### `uppercase(text)`, `lowercase(text)`, `capitalize(text)`

Transform text case.

**Syntax**: 
- `uppercase(text)` - ALL CAPS
- `lowercase(text)` - all lowercase
- `capitalize(text)` - First Letter Caps

**Examples**:
```namel3ss
show text "{{uppercase(user.name)}}"
show text "{{lowercase(user.email)}}"
show text "{{capitalize(product.category)}}"
```

### `trim(text)`

Remove leading and trailing whitespace.

**Syntax**: `trim(text)`

**Examples**:
```namel3ss
filter by: trim(name) != ""
show text "{{trim(user.bio)}}"
```

### `length(text)`

Get the length of a string.

**Syntax**: `length(text)`

**Returns**: `integer`

**Examples**:
```namel3ss
filter by: length(password) >= 8
show text "Characters: {{length(description)}}/500"
```

### `substring(text, start, length?)`

Extract a substring.

**Syntax**: `substring(text, start, length?)`

**Parameters**:
- `text` (string): Source text
- `start` (integer): Starting position (0-indexed)
- `length` (integer, optional): Number of characters to extract

**Examples**:
```namel3ss
show text "Preview: {{substring(article.content, 0, 100)}}..."
show text "Code: {{substring(order.id, 0, 8)}}"
```

### `split(text, delimiter)`

Split a string into an array.

**Syntax**: `split(text, delimiter)`

**Returns**: `list<text>`

**Examples**:
```namel3ss
show list:
    data: split(tags, ",")
    for tag in data:
        show text "{{trim(tag)}}"
```

### `join(list, delimiter)`

Join array elements into a string.

**Syntax**: `join(list, delimiter)`

**Examples**:
```namel3ss
show text "Tags: {{join(product.tags, ', ')}}"
```

---

## JSON Functions

### `parse_json(text)`

Parse a JSON string into an object.

**Syntax**: `parse_json(text)`

**Parameters**:
- `text` (string): JSON string to parse

**Returns**: Object or array

**Examples**:
```namel3ss
dataset "Configs" from postgres "settings":
    schema:
        key: text
        value_json: text

page "Settings" at "/settings":
    for config in Configs:
        show text "{{config.key}}: {{parse_json(config.value_json).setting}}"
```

### `to_json(object)`

Convert an object to JSON string.

**Syntax**: `to_json(object)`

**Parameters**:
- `object` (any): Object to serialize

**Returns**: `text` (JSON string)

**Examples**:
```namel3ss
show text "Debug: {{to_json(user)}}"

action "SaveConfig":
    insert into Settings:
        key: "user_prefs"
        value: to_json(form.preferences)
```

---

## Number Functions

### `round(number, decimals?)`

Round a number to specified decimal places.

**Syntax**: `round(number, decimals?)`

**Parameters**:
- `number` (number): Number to round
- `decimals` (integer, optional): Decimal places (default: 0)

**Examples**:
```namel3ss
show text "Price: ${{round(product.price, 2)}}"
show text "Rating: {{round(review.score, 1)}}"
show text "Count: {{round(avg_visitors)}}"
```

### `abs(number)`

Get absolute value.

**Syntax**: `abs(number)`

**Examples**:
```namel3ss
show text "Difference: {{abs(actual - expected)}}"
```

### `min(a, b)`, `max(a, b)`

Get minimum or maximum of two values.

**Syntax**: 
- `min(a, b)` - Smaller value
- `max(a, b)` - Larger value

**Examples**:
```namel3ss
show text "Price: ${{max(product.price, 0)}}"
show text "Discount: ${{min(discount, max_discount)}}"
```

### `floor(number)`, `ceil(number)`

Round down or up to nearest integer.

**Syntax**:
- `floor(number)` - Round down
- `ceil(number)` - Round up

**Examples**:
```namel3ss
show text "Pages: {{ceil(total_items / page_size)}}"
show text "Complete days: {{floor(hours_worked / 24)}}"
```

---

## Array/List Functions

### `count(list)`

Count items in an array.

**Syntax**: `count(list)`

**Returns**: `integer`

**Examples**:
```namel3ss
show text "{{count(user.orders)}} orders"
filter by: count(tags) > 0
```

### `first(list)`, `last(list)`

Get first or last element.

**Syntax**:
- `first(list)` - First element
- `last(list)` - Last element

**Examples**:
```namel3ss
show text "Latest: {{first(recent_orders).title}}"
show text "Oldest: {{last(orders).date}}"
```

### `contains(list, value)`

Check if array contains a value.

**Syntax**: `contains(list, value)`

**Returns**: `boolean`

**Examples**:
```namel3ss
filter by: contains(permissions, "admin")
if contains(user.roles, "editor"):
    show button "Edit"
```

---

## Conditional Functions

### `if_then_else(condition, true_value, false_value)`

Conditional expression (alternative to ternary operators).

**Syntax**: `if_then_else(condition, true_value, false_value)`

**Parameters**:
- `condition` (boolean): Condition to evaluate
- `true_value` (any): Value if true
- `false_value` (any): Value if false

**Note**: While this function exists, prefer using `if/else` blocks in page statements for clarity.

**Examples**:
```namel3ss
show text "Status: {{if_then_else(order.paid, 'Paid', 'Pending')}}"
show text:
    text: "{{product.name}}"
    style: if_then_else(product.in_stock, "text-success", "text-muted")
```

### `coalesce(value1, value2, ...)`

Return first non-null value.

**Syntax**: `coalesce(value1, value2, ...)`

**Examples**:
```namel3ss
show text "Name: {{coalesce(user.display_name, user.username, 'Anonymous')}}"
show text "Phone: {{coalesce(user.phone, 'N/A')}}"
```

---

## Aggregation Functions

Used in dataset `aggregate` clauses:

### `sum(field)`

Sum of all values.

**Syntax**: `sum(field)`

**Examples**:
```namel3ss
dataset "TotalRevenue" from postgres "orders":
    aggregate:
        total: sum(amount)
```

### `avg(field)`

Average of all values.

**Syntax**: `avg(field)`

**Examples**:
```namel3ss
dataset "AverageScore" from postgres "reviews":
    group by: product_id
    aggregate:
        avg_rating: avg(score)
```

### `count(field)`

Count of rows.

**Syntax**: `count(field)`

**Examples**:
```namel3ss
dataset "UserCounts" from postgres "users":
    group by: status
    aggregate:
        count: count(id)
```

### `min(field)`, `max(field)`

Minimum or maximum value.

**Syntax**:
- `min(field)` - Smallest value
- `max(field)` - Largest value

**Examples**:
```namel3ss
dataset "PriceRange" from postgres "products":
    group by: category
    aggregate:
        lowest: min(price)
        highest: max(price)
```

---

## Context Functions

### `route_param(name)`

Get URL route parameter value.

**Syntax**: `route_param(name)`

**Alternative**: `ctx.route.params.name`

**Examples**:
```namel3ss
page "ProductDetail" at "/products/{id}":
    dataset "Product" from postgres "products":
        filter by: id == route_param("id")
    
    # Or use context directly:
    filter by: id == ctx.route.params.id
```

### `query_param(name, default?)`

Get URL query parameter.

**Syntax**: `query_param(name, default?)`

**Alternative**: `ctx.query.name`

**Examples**:
```namel3ss
page "Search" at "/search":
    dataset "Results" from postgres "products":
        filter by: name contains query_param("q", "")
    
    # Or use context:
    filter by: name contains coalesce(ctx.query.q, "")
```

---

## Validation Functions

### `is_email(text)`

Check if string is valid email format.

**Syntax**: `is_email(text)`

**Returns**: `boolean`

**Examples**:
```namel3ss
show form "ContactForm":
    fields:
        email:
            type: "email"
    validation:
        custom: is_email(form.email)
        message: "Invalid email address"
```

### `is_url(text)`

Check if string is valid URL format.

**Syntax**: `is_url(text)`

**Returns**: `boolean`

### `is_numeric(text)`

Check if string contains only numbers.

**Syntax**: `is_numeric(text)`

**Returns**: `boolean`

---

## Usage Examples

### Example 1: Formatted Dashboard

```namel3ss
dataset "DashboardData" from postgres "orders":
    schema:
        total_orders: integer
        total_revenue: decimal
        avg_order_value: decimal
        last_order_time: timestamp
    aggregate:
        total_orders: count(id)
        total_revenue: sum(total)
        avg_order_value: avg(total)
        last_order_time: max(created_at)

page "Dashboard" at "/":
    for data in DashboardData:
        show card "Overview":
            show text "Total Orders: {{format(data.total_orders, '0,0')}}"
            show text "Revenue: {{format(data.total_revenue, '$0,0.00')}}"
            show text "Avg Order: {{format(data.avg_order_value, '$0.00')}}"
            show text "Last Order: {{format_timestamp(data.last_order_time, 'relative')}}"
```

### Example 2: Search with Formatting

```namel3ss
dataset "ProductSearch" from postgres "products":
    schema:
        id: uuid
        name: text
        price: decimal
        created_at: timestamp
    filter by: lowercase(name) contains lowercase(ctx.query.search)
    order by: created_at desc

page "SearchResults" at "/search":
    show text "Showing {{count(ProductSearch)}} results"
    
    for product in ProductSearch:
        show card:
            show text:
                text: "{{capitalize(product.name)}}"
                style: "font-weight: bold"
            show text "{{format(product.price, '$0,0.00')}}"
            show text "Added {{format_timestamp(product.created_at, 'relative')}}"
```

### Example 3: Data Processing

```namel3ss
dataset "ProcessedJobs" from postgres "jobs":
    schema:
        id: uuid
        filename: text
        status: text
        duration: float
        created_at: timestamp
    filter by: status == "completed"
    filter by: created_at > now() - days(7)

page "JobReport" at "/jobs/report":
    for job in ProcessedJobs:
        show card:
            show text "File: {{uppercase(substring(job.filename, 0, 50))}}"
            show text "Duration: {{format_duration(job.duration)}}"
            show text "Completed: {{format_timestamp(job.created_at, 'medium')}}"
```

---

## Function Availability

| Context | Available Functions |
|---------|-------------------|
| Dataset `filter by:` | All functions except aggregations |
| Dataset `aggregate:` | Aggregation functions (sum, avg, count, min, max) |
| Page templates `{{ }}` | All functions |
| Form validation | Validation functions, comparison functions |
| Expressions | All functions except aggregations |

---

## Notes

### Ternary Operators Not Supported

❌ **Don't use**: `condition ? value1 : value2`

✅ **Use instead**:
```namel3ss
# In templates
{{if_then_else(condition, value1, value2)}}

# In pages
if condition:
    show text "Value 1"
else:
    show text "Value 2"
```

### String Interpolation

Use `{{ }}` for expressions in text:

```namel3ss
show text "Hello {{user.name}}, you have {{count(notifications)}} new messages"
show text "Total: {{format(sum(items.price), '$0,0.00')}}"
```

---

## Related Documentation

- [EXPRESSIONS_REFERENCE.md](./EXPRESSIONS_REFERENCE.md) - Expression syntax
- [DATA_MODELS_GUIDE.md](./DATA_MODELS_GUIDE.md) - Dataset operations
- [QUERIES_AND_DATASETS.md](./QUERIES_AND_DATASETS.md) - Filtering and aggregation
- [FORMS_REFERENCE.md](./FORMS_REFERENCE.md) - Form validation

---

**Last Updated**: November 28, 2025
