# Namel3ss Control Flow Quick Reference

## If/Else Conditionals

### Basic If Block
```n3
if condition:
  statement1
  statement2
```

### If/Else Block
```n3
if condition:
  statement1
else:
  statement2
```

### Supported Conditions
Conditions are currently stored as raw expressions. Common patterns:
- Equality: `user.role == "admin"`
- Inequality: `status != "pending"`
- Comparison: `count > 100`, `price <= 50`
- Any valid expression string

## For Loops

### Loop Over Dataset
```n3
for variable in dataset dataset_name:
  statement1
  statement2
```

Example:
```n3
dataset "active_users" from table users:
  filter by: status == "active"

page "Users" at "/users":
  for user in dataset active_users:
    show text "{user.name} - {user.email}"
```

### Loop Over Table
```n3
for variable in table table_name:
  statement1
  statement2
```

Example:
```n3
page "Orders" at "/orders":
  for order in table orders:
    show text "Order #{order.id}"
```

## Nesting Control Flow

Control flow statements can be nested:

```n3
for item in table items:
  if item.status == "active":
    show text "{item.name} is active"
  else:
    show text "{item.name} is inactive"
```

## Complete Example

```n3
app "Store Dashboard".

dataset "premium_orders" from table orders:
  filter by: amount > 100

page "Dashboard" at "/":
  show text "Welcome"
  
  # Conditional greeting
  if user.role == "admin":
    show text "Admin Panel"
    show table "All Orders" from table orders
  else:
    show text "Your Orders"
  
  # Loop through premium orders
  show text "Premium Orders"
  for order in dataset premium_orders:
    show text "Order #{order.id}: ${order.amount}"
    
    # Nested condition in loop
    if order.status == "shipped":
      show text "✓ Shipped"
    else:
      show text "⏳ Processing"
```

## Indentation Rules

- Control flow blocks must be indented (like Python)
- Use consistent spacing (2 or 4 spaces recommended)
- Statements at the same level must have the same indentation
- `else:` must be at the same indentation as its matching `if`
- Colons (`:`) are required after if, else, and for headers

## Error Handling

The parser provides helpful error messages:

```
Syntax error on line 5: Expected ':' after if condition
if user.role == "admin"
```

```
Syntax error on line 8: Expected at least one statement in if block body
if user.role == "admin":
else:
```

## Tips

1. **Use descriptive variable names** in for loops: `for order in...` not `for x in...`
2. **Keep conditions simple** for now (complex expressions will be supported later)
3. **Indent consistently** to avoid parsing errors
4. **Test conditions** - remember both if and else branches render in static preview
5. **Combine with existing statements** - any `show text`, `show table`, etc. works inside control flow

## What's Generated

### Frontend (HTML)
- If/else blocks show both branches with colored borders for preview
- For loops render 3 sample iterations
- Nested structures maintain visual hierarchy

### Backend
- Control flow structures are recognized but don't yet affect route generation
- Dataset and table sources work as expected
- Future versions will support conditional data fetching
