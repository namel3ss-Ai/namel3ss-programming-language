# UI Components & Styling Guide

This guide covers Namel3ss UI components, styling syntax, conditional rendering, list iteration, and component composition patterns.

## Table of Contents

1. [Component Basics](#component-basics)
2. [Styling & CSS Mapping](#styling--css-mapping)
3. [Conditional Rendering](#conditional-rendering)
4. [List Iteration](#list-iteration)
5. [Component Nesting](#component-nesting)
6. [Fixed Positioning & Chat Widgets](#fixed-positioning--chat-widgets)
7. [Reactive State & Updates](#reactive-state--updates)
8. [Complete Examples](#complete-examples)

---

## Component Basics

Namel3ss provides a rich set of UI components that compile to professional React components:

### Basic Components

```namel3ss
page "Home" at "/" {
    show text "Welcome" {
        style: {
            fontSize: "24px",
            fontWeight: "bold",
            color: "#2563eb"
        }
    }
    
    show form "Contact" {
        fields: [
            {name: "email", type: "email", required: true},
            {name: "message", type: "textarea", required: true}
        ]
        on_submit: {
            run_chain: "ProcessContact"
            show_toast: "Message sent!"
        }
    }
    
    show list "Features" {
        items: [
            {title: "Fast", description: "Lightning fast performance"},
            {title: "Secure", description: "Enterprise-grade security"},
            {title: "Scalable", description: "Grows with your business"}
        ]
    }
}
```

### Data-Bound Components

```namel3ss
page "Dashboard" at "/dashboard" {
    show data_table "Recent Orders" {
        source: "orders"
        columns: ["id", "customer_name", "total", "status"]
        page_size: 20
        enable_search: true
        sortable: true
    }
    
    show data_list "Notifications" {
        source: "notifications"
        item: {
            title: "{user.name}"
            subtitle: "{message}"
            badge: "{status}"
            icon: "{type}"
        }
    }
}
```

---

## Styling & CSS Mapping

### Basic Style Syntax

Namel3ss supports CSS properties via the `style:` block:

```namel3ss
show text "Styled Text" {
    style: {
        fontSize: "18px",
        fontWeight: "bold",
        color: "#2563eb",
        backgroundColor: "#f1f5f9",
        padding: "16px",
        borderRadius: "8px",
        textAlign: "center"
    }
}
```

### CSS Property Mapping

| Namel3ss Property | CSS Property | Example Values |
|------------------|--------------|----------------|
| `fontSize` | `font-size` | `"16px"`, `"1.2rem"`, `"large"` |
| `fontWeight` | `font-weight` | `"bold"`, `"400"`, `"semibold"` |
| `color` | `color` | `"#2563eb"`, `"blue"`, `"var(--primary)"` |
| `backgroundColor` | `background-color` | `"#f1f5f9"`, `"transparent"` |
| `padding` | `padding` | `"16px"`, `"1rem 2rem"`, `"0.5rem"` |
| `margin` | `margin` | `"8px 0"`, `"1rem"`, `"auto"` |
| `borderRadius` | `border-radius` | `"8px"`, `"50%"`, `"0.5rem"` |
| `border` | `border` | `"1px solid #e5e7eb"` |
| `textAlign` | `text-align` | `"center"`, `"left"`, `"right"` |
| `display` | `display` | `"flex"`, `"grid"`, `"none"` |
| `flexDirection` | `flex-direction` | `"column"`, `"row"` |
| `justifyContent` | `justify-content` | `"center"`, `"space-between"` |
| `alignItems` | `align-items` | `"center"`, `"flex-start"` |
| `gap` | `gap` | `"16px"`, `"1rem"` |
| `width` | `width` | `"100%"`, `"300px"`, `"auto"` |
| `height` | `height` | `"200px"`, `"100vh"`, `"auto"` |
| `position` | `position` | `"fixed"`, `"absolute"`, `"relative"` |
| `top`, `right`, `bottom`, `left` | positioning | `"16px"`, `"0"`, `"50%"` |
| `zIndex` | `z-index` | `"1000"`, `"999"` |

### Unit Handling

- **Strings with units**: `"16px"`, `"1.5rem"`, `"50%"` → passed through as-is
- **Numbers**: `16`, `1.5` → interpreted as `px` values
- **Special values**: `"auto"`, `"inherit"`, `"initial"` → passed through

### Layout Examples

```namel3ss
// Flexbox layout
show div {
    style: {
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "1rem",
        padding: "1rem"
    }
    
    children: [
        show text "Left content",
        show text "Right content"
    ]
}

// Grid layout
show div {
    style: {
        display: "grid",
        gridTemplateColumns: "1fr 1fr 1fr",
        gap: "1rem",
        padding: "1rem"
    }
    
    children: [
        show card "Card 1",
        show card "Card 2", 
        show card "Card 3"
    ]
}
```

---

## Conditional Rendering

Use `if` statements to conditionally render components based on data or expressions:

### Basic Conditionals

```namel3ss
page "Profile" at "/profile" {
    if user.is_premium:
        show text "Welcome, Premium Member!" {
            style: {color: "#10b981", fontWeight: "bold"}
        }
    else:
        show text "Upgrade to Premium" {
            style: {color: "#6b7280"}
        }
    
    if user.unread_messages > 0:
        show badge "{user.unread_messages} new messages" {
            style: {backgroundColor: "#ef4444", color: "white"}
        }
}
```

### Conditional Styling

```namel3ss
// Different styles based on message role
show list "Chat Messages" {
    from: memory.chat_history
    item as message:
        if message.role == "user":
            show text message.content {
                style: {
                    alignSelf: "flex-end",
                    backgroundColor: "#2563eb",
                    color: "white",
                    borderRadius: "18px 18px 4px 18px",
                    padding: "12px 16px",
                    maxWidth: "70%",
                    marginBottom: "8px"
                }
            }
        elif message.role == "assistant":
            show text message.content {
                style: {
                    alignSelf: "flex-start",
                    backgroundColor: "#f1f5f9",
                    color: "#1f2937",
                    borderRadius: "18px 18px 18px 4px",
                    padding: "12px 16px",
                    maxWidth: "70%",
                    marginBottom: "8px"
                }
            }
        else:
            show text message.content {
                style: {
                    alignSelf: "center",
                    backgroundColor: "#fef3c7",
                    color: "#92400e",
                    borderRadius: "8px",
                    padding: "8px 12px",
                    fontSize: "14px",
                    fontStyle: "italic"
                }
            }
}
```

### Complex Conditionals

```namel3ss
page "Dashboard" at "/dashboard" {
    if user.role == "admin":
        show text "Admin Dashboard" {
            style: {fontSize: "24px", fontWeight: "bold"}
        }
        
        show data_table "All Users" {
            source: "all_users"
            columns: ["name", "email", "role", "last_login"]
        }
        
    elif user.role == "manager":
        show text "Manager Dashboard"
        
        show data_table "Team Members" {
            source: "team_users"
            filter_by: "team_id == user.team_id"
        }
        
    else:
        show text "User Dashboard"
        
        show card "Your Stats" {
            data: user_stats
        }
}
```

---

## List Iteration

Render collections of data using the `show list` with `from:` and `item` syntax:

### Basic List Iteration

```namel3ss
// Iterate over static data
show list "Product Features" {
    from: [
        {name: "Fast", icon: "⚡", description: "Lightning fast"},
        {name: "Secure", icon: "🔒", description: "Bank-level security"},
        {name: "Scalable", icon: "📈", description: "Grows with you"}
    ]
    item as feature:
        show card feature.name {
            icon: feature.icon
            description: feature.description
        }
}

// Iterate over memory
show list "Chat History" {
    from: memory.chat_history
    item as message:
        show div {
            style: {
                display: "flex",
                flexDirection: message.role == "user" ? "row-reverse" : "row",
                marginBottom: "1rem"
            }
            
            children: [
                show text message.content {
                    style: {
                        padding: "12px 16px",
                        borderRadius: "18px",
                        backgroundColor: message.role == "user" ? "#2563eb" : "#f1f5f9",
                        color: message.role == "user" ? "white" : "#1f2937",
                        maxWidth: "70%"
                    }
                }
            ]
        }
}
```

### List with Filtering

```namel3ss
// Show only unread notifications
show list "Unread Notifications" {
    from: notifications
    filter: "is_read == false"
    item as notification:
        show card notification.title {
            subtitle: notification.message
            badge: notification.type
            style: {
                borderLeft: "4px solid #ef4444"
            }
        }
}

// Show recent orders (last 10)
show list "Recent Orders" {
    from: orders
    limit: 10
    sort_by: "created_at desc"
    item as order:
        show div {
            style: {
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "1rem",
                borderBottom: "1px solid #e5e7eb"
            }
            
            children: [
                show text "Order #{order.id}",
                show text "${order.total}" {
                    style: {fontWeight: "bold"}
                },
                show badge order.status
            ]
        }
}
```

### Nested Lists

```namel3ss
// Categories with products
show list "Product Categories" {
    from: categories
    item as category:
        show div {
            style: {marginBottom: "2rem"}
            
            children: [
                show text category.name {
                    style: {fontSize: "20px", fontWeight: "bold", marginBottom: "1rem"}
                },
                
                show list "Products" {
                    from: category.products
                    item as product:
                        show card product.name {
                            price: product.price
                            image: product.image_url
                            style: {display: "inline-block", width: "200px", margin: "0 1rem 1rem 0"}
                        }
                }
            ]
        }
}
```

### Error Handling

```namel3ss
show list "User Messages" {
    from: memory.chat_history
    empty_state: {
        icon: "💬",
        title: "No messages yet",
        message: "Start a conversation to see messages here"
    }
    error_state: {
        icon: "⚠️",
        title: "Failed to load messages",
        message: "Please try again later",
        action_label: "Retry",
        action: "refresh_messages"
    }
    item as message:
        // render message
}
```

---

## Component Nesting

Create complex layouts by nesting components:

### Layout Components

```namel3ss
page "Dashboard" at "/dashboard" {
    show stack {
        direction: "vertical"
        gap: "2rem"
        style: {padding: "2rem"}
        
        children: [
            // Header section
            show div {
                style: {
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                }
                
                children: [
                    show text "Dashboard" {
                        style: {fontSize: "32px", fontWeight: "bold"}
                    },
                    show button "New Item" {
                        variant: "primary"
                        on_click: "create_item_modal"
                    }
                ]
            },
            
            // Stats section
            show grid {
                columns: 3
                gap: "1rem"
                
                children: [
                    show stat_card {
                        title: "Total Users"
                        value: "{stats.total_users}"
                        change: "+12%"
                        trend: "up"
                    },
                    show stat_card {
                        title: "Revenue"
                        value: "${stats.revenue}"
                        change: "+8%"
                        trend: "up"
                    },
                    show stat_card {
                        title: "Conversion"
                        value: "{stats.conversion_rate}%"
                        change: "-2%"
                        trend: "down"
                    }
                ]
            },
            
            // Main content
            show grid {
                columns: "2fr 1fr"
                gap: "2rem"
                
                children: [
                    show data_table "Recent Orders" {
                        source: "orders"
                        columns: ["id", "customer", "total", "status"]
                    },
                    
                    show card "Quick Actions" {
                        children: [
                            show button "Create Order" {on_click: "create_order"},
                            show button "Export Data" {on_click: "export_data"},
                            show button "Settings" {on_click: "open_settings"}
                        ]
                    }
                ]
            }
        ]
    }
}
```

### Component Composition

```namel3ss
// Reusable components using functions
fn user_avatar(user, size = "40px") {
    show div {
        style: {
            width: size,
            height: size,
            borderRadius: "50%",
            backgroundColor: user.avatar ? "transparent" : "#e5e7eb",
            backgroundImage: user.avatar ? "url({user.avatar})" : "none",
            backgroundSize: "cover",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "14px",
            fontWeight: "bold",
            color: "#6b7280"
        }
        
        children: [
            if !user.avatar:
                show text "{user.name[0].toUpperCase()}"
        ]
    }
}

fn message_bubble(message, is_own = false) {
    show div {
        style: {
            display: "flex",
            flexDirection: is_own ? "row-reverse" : "row",
            alignItems: "flex-start",
            gap: "8px",
            marginBottom: "1rem"
        }
        
        children: [
            user_avatar(message.sender, "32px"),
            
            show div {
                style: {
                    backgroundColor: is_own ? "#2563eb" : "#f1f5f9",
                    color: is_own ? "white" : "#1f2937",
                    borderRadius: is_own ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    padding: "12px 16px",
                    maxWidth: "70%"
                }
                
                children: [
                    show text message.content,
                    show text message.timestamp {
                        style: {
                            fontSize: "12px",
                            opacity: "0.7",
                            marginTop: "4px"
                        }
                    }
                ]
            }
        ]
    }
}

// Use composed components
page "Chat" at "/chat/{conversation_id}" {
    show div {
        style: {
            height: "100vh",
            display: "flex",
            flexDirection: "column"
        }
        
        children: [
            // Chat header
            show div {
                style: {
                    padding: "1rem",
                    borderBottom: "1px solid #e5e7eb"
                }
                children: [
                    show text conversation.title {
                        style: {fontSize: "18px", fontWeight: "bold"}
                    }
                ]
            },
            
            // Messages
            show div {
                style: {
                    flex: "1",
                    overflowY: "auto",
                    padding: "1rem"
                }
                
                children: [
                    show list "Messages" {
                        from: conversation.messages
                        item as message:
                            message_bubble(message, message.sender_id == user.id)
                    }
                ]
            },
            
            // Input form
            show form "Send Message" {
                style: {
                    padding: "1rem",
                    borderTop: "1px solid #e5e7eb"
                }
                
                fields: [
                    {
                        name: "message",
                        type: "text",
                        placeholder: "Type a message...",
                        style: {
                            border: "none",
                            outline: "none",
                            flex: "1"
                        }
                    }
                ]
                
                on_submit: {
                    run_chain: "SendMessage",
                    clear_form: true
                }
            }
        ]
    }
}
```

---

## Fixed Positioning & Chat Widgets

Create floating chat widgets and overlays using fixed positioning:

### Floating Chat Bubble

```namel3ss
page "Homepage" at "/" {
    // Main page content
    show text "Welcome to our website!" {
        style: {fontSize: "24px", marginBottom: "2rem"}
    }
    
    show div "Main content area" {
        style: {minHeight: "80vh"}
    }
    
    // Floating chat widget
    show div {
        id: "chat-widget"
        style: {
            position: "fixed",
            bottom: "20px",
            right: "20px",
            zIndex: "1000"
        }
        
        children: [
            // Chat button (when minimized)
            if !chat.is_open:
                show button {
                    style: {
                        width: "60px",
                        height: "60px",
                        borderRadius: "50%",
                        backgroundColor: "#2563eb",
                        border: "none",
                        color: "white",
                        fontSize: "24px",
                        cursor: "pointer",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.15)"
                    }
                    
                    on_click: "toggle_chat"
                    
                    children: [
                        show text "💬"
                    ]
                }
            
            // Chat window (when open)
            else:
                show div {
                    style: {
                        width: "350px",
                        height: "500px",
                        backgroundColor: "white",
                        borderRadius: "12px",
                        boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
                        border: "1px solid #e5e7eb",
                        display: "flex",
                        flexDirection: "column",
                        overflow: "hidden"
                    }
                    
                    children: [
                        // Chat header
                        show div {
                            style: {
                                padding: "16px",
                                backgroundColor: "#2563eb",
                                color: "white",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center"
                            }
                            
                            children: [
                                show text "Chat Support" {
                                    style: {fontWeight: "bold"}
                                },
                                show button "✕" {
                                    style: {
                                        background: "none",
                                        border: "none",
                                        color: "white",
                                        cursor: "pointer",
                                        fontSize: "18px"
                                    }
                                    on_click: "toggle_chat"
                                }
                            ]
                        },
                        
                        // Messages area
                        show div {
                            style: {
                                flex: "1",
                                overflowY: "auto",
                                padding: "16px"
                            }
                            
                            children: [
                                show list "Chat Messages" {
                                    from: memory.chat_history
                                    item as message:
                                        show div {
                                            style: {
                                                display: "flex",
                                                flexDirection: message.role == "user" ? "row-reverse" : "row",
                                                marginBottom: "12px"
                                            }
                                            
                                            children: [
                                                show text message.content {
                                                    style: {
                                                        padding: "8px 12px",
                                                        borderRadius: message.role == "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                                                        backgroundColor: message.role == "user" ? "#2563eb" : "#f1f5f9",
                                                        color: message.role == "user" ? "white" : "#1f2937",
                                                        maxWidth: "80%",
                                                        fontSize: "14px"
                                                    }
                                                }
                                            ]
                                        }
                                }
                            ]
                        },
                        
                        // Input area
                        show form "Chat Input" {
                            style: {
                                padding: "16px",
                                borderTop: "1px solid #e5e7eb"
                            }
                            
                            fields: [
                                {
                                    name: "message",
                                    type: "text",
                                    placeholder: "Type your message...",
                                    style: {
                                        width: "100%",
                                        padding: "8px 12px",
                                        border: "1px solid #e5e7eb",
                                        borderRadius: "6px",
                                        outline: "none"
                                    }
                                }
                            ]
                            
                            on_submit: {
                                run_chain: "ProcessChatMessage",
                                clear_form: true
                            }
                        }
                    ]
                }
        ]
    }
}
```

### Modal Overlays

```namel3ss
// Modal overlay
if modal.is_open:
    show div {
        style: {
            position: "fixed",
            top: "0",
            left: "0",
            right: "0",
            bottom: "0",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: "2000"
        }
        
        on_click: "close_modal"
        
        children: [
            show div {
                style: {
                    backgroundColor: "white",
                    borderRadius: "12px",
                    padding: "24px",
                    maxWidth: "500px",
                    width: "90%",
                    maxHeight: "90%",
                    overflowY: "auto",
                    boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)"
                }
                
                on_click: "event.stopPropagation()"
                
                children: [
                    show text modal.title {
                        style: {fontSize: "24px", fontWeight: "bold", marginBottom: "16px"}
                    },
                    show text modal.content {
                        style: {marginBottom: "24px", lineHeight: "1.5"}
                    },
                    
                    show div {
                        style: {
                            display: "flex",
                            justifyContent: "flex-end",
                            gap: "12px"
                        }
                        
                        children: [
                            show button "Cancel" {
                                variant: "secondary"
                                on_click: "close_modal"
                            },
                            show button modal.confirm_text {
                                variant: "primary"
                                on_click: modal.confirm_action
                            }
                        ]
                    }
                ]
            }
        ]
    }
```

### Sticky Header

```namel3ss
page "Article" at "/article/{id}" {
    // Sticky navigation
    show div {
        style: {
            position: "sticky",
            top: "0",
            backgroundColor: "white",
            borderBottom: "1px solid #e5e7eb",
            padding: "16px 0",
            zIndex: "100"
        }
        
        children: [
            show div {
                style: {
                    maxWidth: "800px",
                    margin: "0 auto",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                }
                
                children: [
                    show text "← Back to Articles" {
                        style: {color: "#2563eb", cursor: "pointer"}
                        on_click: "navigate_back"
                    },
                    
                    show div {
                        style: {display: "flex", gap: "12px"}
                        children: [
                            show button "Share",
                            show button "Bookmark"
                        ]
                    }
                ]
            }
        ]
    }
    
    // Article content
    show div {
        style: {
            maxWidth: "800px",
            margin: "0 auto",
            padding: "32px 16px"
        }
        
        children: [
            show text article.title {
                style: {fontSize: "36px", fontWeight: "bold", marginBottom: "16px"}
            },
            show text article.content {
                style: {lineHeight: "1.7", fontSize: "18px"}
            }
        ]
    }
}
```

---

## Reactive State & Updates

Namel3ss provides several mechanisms for reactive UI updates:

### Auto-Refresh on Memory Changes

```namel3ss
page "Chat" at "/chat" {
    reactive: true  // Enable reactive updates
    
    show list "Messages" {
        from: memory.chat_history
        reactive: true  // Auto-refresh when memory changes
        item as message:
            show div {
                children: [show text message.content]
            }
    }
    
    show form "Send Message" {
        fields: [message]
        on_submit: {
            run_chain: "SendMessage"  // Chain updates memory
            // UI automatically refreshes due to reactive: true
        }
    }
}
```

### Manual Refresh Commands

```namel3ss
page "Dashboard" at "/dashboard" {
    show data_table "Orders" {
        source: "orders"
        refresh_on: ["order_created", "order_updated"]
    }
    
    show button "Refresh Data" {
        on_click: {
            refresh: ["orders", "stats"],
            show_toast: "Data refreshed"
        }
    }
}
```

### Conditional Updates

```namel3ss
page "Status" at "/status" {
    refresh_interval: "5s"  // Poll every 5 seconds
    
    show div {
        style: {
            backgroundColor: system.status == "healthy" ? "#10b981" : "#ef4444",
            color: "white",
            padding: "16px",
            borderRadius: "8px",
            textAlign: "center"
        }
        
        children: [
            show text "System Status: {system.status}",
            
            if system.status != "healthy":
                show text system.error_message {
                    style: {marginTop: "8px", fontSize: "14px"}
                }
        ]
    }
}
```

### Real-time Updates

```namel3ss
page "LiveChat" at "/chat/{room_id}" {
    realtime: true  // Enable WebSocket connection
    
    show div {
        children: [
            show list "Messages" {
                from: memory.chat_messages
                realtime_source: "chat_room_{room_id}"
                item as message:
                    show div {
                        style: {
                            padding: "8px 12px",
                            margin: "4px 0",
                            borderRadius: "8px",
                            backgroundColor: message.is_mine ? "#dbeafe" : "#f3f4f6"
                        }
                        children: [
                            show text message.content,
                            show text message.timestamp {
                                style: {fontSize: "12px", color: "#6b7280"}
                            }
                        ]
                    }
            },
            
            show form "Send Message" {
                fields: [message_text]
                on_submit: {
                    run_chain: "SendChatMessage",
                    clear_form: true,
                    broadcast: "chat_room_{room_id}"  // Real-time broadcast
                }
                style: {
                    position: "sticky",
                    bottom: "0px",
                    backgroundColor: "white",
                    borderTop: "1px solid #e5e7eb"
                }
            }
        ]
    }
}
```

### Form Processing & Validation

```namel3ss
page "ContactUs" at "/contact" {
    show form "Contact Form" {
        fields: [
            field name: {
                type: "text",
                required: true,
                validation: {pattern: "[A-Za-z ]+", message: "Name must only contain letters"}
            },
            field email: {
                type: "email", 
                required: true,
                validation: {message: "Please enter a valid email"}
            },
            field message: {
                type: "textarea",
                required: true,
                validation: {min_length: 10, message: "Message must be at least 10 characters"}
            }
        ]
        
        on_submit: {
            run_chain: "ProcessContact",
            show_toast: "Thank you! We'll be in touch soon.",
            redirect: "/thank-you"
        }
        
        on_error: {
            show_toast: "Please fix the errors above"
        }
        
        style: {
            maxWidth: "500px",
            margin: "0 auto",
            padding: "24px"
        }
    }
}
```

### Enhanced List Iteration with State

```namel3ss
page "TodoApp" at "/todos" {
    reactive: true
    
    show list "Todo Items" {
        from: memory.todos
        item as todo:
            show div {
                style: {
                    display: "flex",
                    alignItems: "center",
                    padding: "12px",
                    borderBottom: "1px solid #e5e7eb",
                    backgroundColor: todo.completed ? "#f9fafb" : "white"
                }
                
                children: [
                    show checkbox {
                        checked: todo.completed,
                        on_change: {
                            update_memory: "todos[{todo.id}].completed" = !todo.completed,
                            refresh: true
                        }
                        style: {marginRight: "12px"}
                    },
                    
                    show text todo.text {
                        style: {
                            flex: "1",
                            textDecoration: todo.completed ? "line-through" : "none",
                            color: todo.completed ? "#6b7280" : "#111827"
                        }
                    },
                    
                    show button "Delete" {
                        on_click: {
                            run_chain: "DeleteTodo",
                            payload: {todo_id: todo.id},
                            refresh: true
                        }
                        style: {
                            color: "#dc2626",
                            background: "none",
                            border: "none",
                            cursor: "pointer"
                        }
                    }
                ]
            }
        
        empty_state:
            show text "No todos yet! Add one below." {
                style: {textAlign: "center", color: "#6b7280", padding: "24px"}
            }
    }
    
    show form "Add Todo" {
        fields: [
            field text: {placeholder: "What needs to be done?"}
        ]
        
        on_submit: {
            run_chain: "AddTodo",
            clear_form: true,
            refresh: true
        }
        
        style: {
            position: "sticky",
            bottom: "0px",
            padding: "16px",
            backgroundColor: "white",
            borderTop: "1px solid #e5e7eb"
        }
    }
}
```

---

## Complete Examples

### Customer Support Chat Widget

```namel3ss
app "Support Chat Widget" {
    realtime: true
}

memory "chat_history" {
    scope: "session"
    kind: "list"
    max_items: 100
}

chain "ProcessChatMessage" {
    input: {message: text}
    
    steps: [
        step "save_user_message" {
            kind: "memory_write"
            target: "chat_history"
            options: {
                value: {
                    role: "user",
                    content: input.message,
                    timestamp: context.now
                }
            }
        },
        
        step "generate_response" {
            kind: "prompt"
            target: "SupportAssistant"
            options: {
                message: input.message,
                history: context.chat_history
            }
        },
        
        step "save_assistant_response" {
            kind: "memory_write"
            target: "chat_history"
            options: {
                value: {
                    role: "assistant",
                    content: steps.generate_response.output.response,
                    timestamp: context.now
                }
            }
        }
    ]
}

page "Homepage" at "/" {
    reactive: true
    
    // Main page content
    show text "Welcome to our website!" {
        style: {fontSize: "32px", textAlign: "center", padding: "4rem 0"}
    }
    
    // Floating chat widget
    show div {
        id: "chat-widget"
        style: {
            position: "fixed",
            bottom: "20px",
            right: "20px",
            zIndex: "1000"
        }
        
        children: [
            if !state.chat_open:
                show button {
                    style: {
                        width: "60px",
                        height: "60px",
                        borderRadius: "50%",
                        backgroundColor: "#2563eb",
                        border: "none",
                        color: "white",
                        fontSize: "24px",
                        cursor: "pointer",
                        boxShadow: "0 4px 12px rgba(37,99,235,0.3)",
                        transition: "all 0.2s"
                    }
                    
                    on_click: {
                        set_state: {chat_open: true}
                    }
                    
                    children: [show text "💬"]
                }
            else:
                show div {
                    style: {
                        width: "400px",
                        height: "600px",
                        backgroundColor: "white",
                        borderRadius: "16px",
                        boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
                        border: "1px solid #e5e7eb",
                        display: "flex",
                        flexDirection: "column",
                        overflow: "hidden"
                    }
                    
                    children: [
                        // Header
                        show div {
                            style: {
                                padding: "20px",
                                backgroundColor: "#2563eb",
                                color: "white",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center"
                            }
                            
                            children: [
                                show div {
                                    children: [
                                        show text "Support Chat" {
                                            style: {fontWeight: "bold", fontSize: "16px"}
                                        },
                                        show text "We're here to help!" {
                                            style: {fontSize: "14px", opacity: "0.9", marginTop: "2px"}
                                        }
                                    ]
                                },
                                
                                show button "✕" {
                                    style: {
                                        background: "none",
                                        border: "none",
                                        color: "white",
                                        cursor: "pointer",
                                        fontSize: "20px",
                                        padding: "4px"
                                    }
                                    
                                    on_click: {
                                        set_state: {chat_open: false}
                                    }
                                }
                            ]
                        },
                        
                        // Messages
                        show div {
                            style: {
                                flex: "1",
                                overflowY: "auto",
                                padding: "20px",
                                backgroundColor: "#fafafa"
                            }
                            
                            children: [
                                // Welcome message if no history
                                if count(memory.chat_history) == 0:
                                    show div {
                                        style: {
                                            backgroundColor: "#f1f5f9",
                                            padding: "16px",
                                            borderRadius: "12px",
                                            marginBottom: "16px",
                                            textAlign: "center"
                                        }
                                        
                                        children: [
                                            show text "👋 Hi there! How can we help you today?"
                                        ]
                                    }
                                
                                // Chat history
                                show list "Chat Messages" {
                                    from: memory.chat_history
                                    reactive: true
                                    item as message:
                                        show div {
                                            style: {
                                                display: "flex",
                                                flexDirection: message.role == "user" ? "row-reverse" : "row",
                                                marginBottom: "16px"
                                            }
                                            
                                            children: [
                                                show div {
                                                    style: {
                                                        maxWidth: "85%",
                                                        padding: "12px 16px",
                                                        borderRadius: message.role == "user" ? "20px 20px 4px 20px" : "20px 20px 20px 4px",
                                                        backgroundColor: message.role == "user" ? "#2563eb" : "white",
                                                        color: message.role == "user" ? "white" : "#374151",
                                                        boxShadow: message.role == "user" ? "none" : "0 1px 2px rgba(0, 0, 0, 0.1)",
                                                        lineHeight: "1.4"
                                                    }
                                                    
                                                    children: [
                                                        show text message.content
                                                    ]
                                                }
                                            ]
                                        }
                                }
                            ]
                        },
                        
                        // Input area
                        show div {
                            style: {
                                padding: "20px",
                                backgroundColor: "white",
                                borderTop: "1px solid #e5e7eb"
                            }
                            
                            children: [
                                show form "Chat Input" {
                                    style: {
                                        display: "flex",
                                        gap: "12px"
                                    }
                                    
                                    fields: [
                                        {
                                            name: "message",
                                            type: "text",
                                            placeholder: "Type your message...",
                                            required: true,
                                            style: {
                                                flex: "1",
                                                padding: "12px 16px",
                                                border: "1px solid #d1d5db",
                                                borderRadius: "24px",
                                                outline: "none",
                                                fontSize: "14px"
                                            }
                                        }
                                    ]
                                    
                                    submit_button: {
                                        text: "Send",
                                        style: {
                                            padding: "12px 20px",
                                            backgroundColor: "#2563eb",
                                            color: "white",
                                            border: "none",
                                            borderRadius: "24px",
                                            cursor: "pointer",
                                            fontSize: "14px",
                                            fontWeight: "bold"
                                        }
                                    }
                                    
                                    on_submit: {
                                        run_chain: "ProcessChatMessage",
                                        clear_form: true
                                    }
                                }
                            ]
                        }
                    ]
                }
        ]
    }
}
```

This comprehensive guide provides the foundation for building sophisticated UI components in Namel3ss, with particular focus on the patterns needed for chat applications and real-time interfaces.