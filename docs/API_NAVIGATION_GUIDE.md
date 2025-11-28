# API Integration & Navigation Guide

This guide covers API integration, programmatic navigation, template functions, and embeddable widget patterns in Namel3ss.

## API Response Formatting

### Smart API Integration

```namel3ss
page "APIExplorer" at "/api-explorer" {
    memory: endpoint = "", response_data = null, loading = false
    
    show form "API Request" {
        fields: [
            field endpoint: {
                type: "text",
                placeholder: "https://api.example.com/users",
                label: "API Endpoint"
            },
            field method: {
                type: "select",
                options: ["GET", "POST", "PUT", "DELETE"],
                default: "GET"
            },
            field headers: {
                type: "textarea",
                placeholder: '{"Authorization": "Bearer token"}',
                label: "Headers (JSON)"
            }
        ]
        
        on_submit: {
            update_memory: "loading" = true,
            run_chain: "CallAPI",
            update_memory: "response_data" = result.data,
            update_memory: "loading" = false
        }
    }
    
    if loading:
        show div "Loading..." {
            style: {textAlign: "center", padding: "32px"}
        }
    
    if response_data:
        show div "API Response" {
            children: [
                show text "Response Status: {response_data.status}" {
                    style: {
                        color: response_data.status < 400 ? "#10b981" : "#dc2626",
                        fontWeight: "600",
                        marginBottom: "16px"
                    }
                },
                
                show code_block response_data.body {
                    language: "json",
                    style: {
                        backgroundColor: "#f8fafc",
                        border: "1px solid #e2e8f0",
                        borderRadius: "8px",
                        padding: "16px",
                        overflow: "auto",
                        maxHeight: "400px"
                    }
                },
                
                // Auto-format common response patterns
                if response_data.body.data && response_data.body.data.length > 0:
                    show data_table "Parsed Data" {
                        data: response_data.body.data,
                        auto_columns: true,
                        pagination: true,
                        export_csv: true
                    }
            ]
        }
}
```

### Chain-Based API Processing

```namel3ss
chain "CallAPI" {
    input: {endpoint, method, headers}
    
    step validate:
        if !endpoint.startsWith("https://"):
            throw "Only HTTPS endpoints are allowed"
    
    step prepare_request:
        headers_obj = headers ? JSON.parse(headers) : {}
        headers_obj["User-Agent"] = "Namel3ss/1.0"
    
    step make_request:
        response = fetch(endpoint, {
            method: method,
            headers: headers_obj
        })
    
    step format_response:
        formatted = {
            status: response.status,
            headers: response.headers,
            body: response.json()
        }
        
        // Smart data extraction for common APIs
        if formatted.body.items:
            formatted.parsed_list = formatted.body.items
        elif formatted.body.data && Array.isArray(formatted.body.data):
            formatted.parsed_list = formatted.body.data
        elif formatted.body.results:
            formatted.parsed_list = formatted.body.results
    
    output: formatted
}
```

## Programmatic Navigation

### Dynamic Route Navigation

```namel3ss
page "ProductCatalog" at "/products" {
    memory: selected_category = "all", search_term = ""
    
    show div "Navigation Controls" {
        children: [
            show nav_tabs {
                tabs: [
                    {id: "all", label: "All Products", route: "/products"},
                    {id: "electronics", label: "Electronics", route: "/products/electronics"},
                    {id: "clothing", label: "Clothing", route: "/products/clothing"},
                    {id: "books", label: "Books", route: "/products/books"}
                ],
                active: selected_category,
                on_tab_change: {
                    update_memory: "selected_category" = tab.id,
                    navigate: tab.route,
                    update_url: true  // Update browser URL without page reload
                }
            },
            
            show search_input {
                placeholder: "Search products...",
                value: search_term,
                on_change: {
                    update_memory: "search_term" = value,
                    // Update URL with search params
                    update_url: "/products?category={selected_category}&search={value}"
                }
            }
        ]
    }
    
    show product_grid {
        category: selected_category,
        search: search_term,
        on_product_click: {
            navigate: "/product/{product.id}",
            track_event: "product_viewed",
            store_breadcrumb: product.name
        }
    }
}
```

### Modal and Overlay Navigation

```namel3ss
page "Dashboard" at "/dashboard" {
    memory: modal_open = false, modal_content = null
    
    show div "Action Bar" {
        children: [
            show button "Create New Item" {
                on_click: {
                    update_memory: "modal_content" = "create_form",
                    update_memory: "modal_open" = true,
                    update_url: "/dashboard?modal=create",  // For bookmarkable URLs
                    focus_element: "#item_name_input"
                }
            },
            
            show button "Import Data" {
                on_click: {
                    update_memory: "modal_content" = "import_wizard",
                    update_memory: "modal_open" = true,
                    update_url: "/dashboard?modal=import"
                }
            }
        ]
    }
    
    // Modal overlay
    if modal_open:
        show modal {
            backdrop_close: true,
            on_close: {
                update_memory: "modal_open" = false,
                update_memory: "modal_content" = null,
                navigate_back: true  // Return to previous URL
            }
            
            content: {
                if modal_content == "create_form":
                    show form "Create Item" {
                        fields: [
                            field name: {id: "item_name_input", focus: true},
                            field description: {type: "textarea"}
                        ]
                        on_submit: {
                            run_chain: "CreateItem",
                            update_memory: "modal_open" = false,
                            navigate: "/item/{result.item_id}",
                            show_toast: "Item created successfully"
                        }
                    }
                
                elif modal_content == "import_wizard":
                    show multi_step_wizard {
                        steps: ["upload", "mapping", "preview", "import"],
                        on_complete: {
                            update_memory: "modal_open" = false,
                            refresh_page: true,
                            show_toast: "Import completed"
                        }
                    }
            }
        }
}
```

## Template Functions & Reusable Components

### Custom Template Functions

```namel3ss
// Define reusable template functions
template user_avatar(user, size = "medium") {
    show img {
        src: user.avatar_url || "/default-avatar.png",
        alt: user.name,
        style: {
            width: size == "small" ? "32px" : size == "large" ? "64px" : "48px",
            height: size == "small" ? "32px" : size == "large" ? "64px" : "48px",
            borderRadius: "50%",
            objectFit: "cover",
            border: user.is_online ? "3px solid #10b981" : "3px solid #e5e7eb"
        }
    }
}

template message_bubble(message, is_own = false) {
    show div {
        style: {
            display: "flex",
            justifyContent: is_own ? "flex-end" : "flex-start",
            marginBottom: "12px"
        }
        
        children: [
            if !is_own:
                user_avatar(message.sender, "small"),
                
            show div {
                style: {
                    maxWidth: "70%",
                    marginLeft: is_own ? "0" : "12px",
                    marginRight: is_own ? "12px" : "0"
                }
                
                children: [
                    if !is_own:
                        show text message.sender.name {
                            style: {fontSize: "12px", color: "#6b7280", marginBottom: "4px"}
                        }
                    
                    show div {
                        style: {
                            padding: "8px 12px",
                            borderRadius: "12px",
                            backgroundColor: is_own ? "#3b82f6" : "#f3f4f6",
                            color: is_own ? "white" : "#111827"
                        }
                        children: [
                            show text message.content,
                            show text format_timestamp(message.created_at) {
                                style: {fontSize: "10px", opacity: "0.7", marginTop: "4px"}
                            }
                        ]
                    }
                ]
            },
            
            if is_own:
                user_avatar(message.sender, "small")
        ]
    }
}

template status_badge(status, label = null) {
    show span {
        style: {
            display: "inline-block",
            padding: "4px 8px",
            borderRadius: "12px",
            fontSize: "12px",
            fontWeight: "600",
            backgroundColor: {
                "active": "#dcfce7",
                "pending": "#fef3c7", 
                "error": "#fef2f2",
                "default": "#f3f4f6"
            }[status] || "#f3f4f6",
            color: {
                "active": "#166534",
                "pending": "#92400e",
                "error": "#991b1b", 
                "default": "#374151"
            }[status] || "#374151"
        }
        children: [
            show text label || status.toUpperCase()
        ]
    }
}
```

### Using Template Functions

```namel3ss
page "TeamChat" at "/team/{team_id}" {
    realtime: true
    
    show div "Chat Messages" {
        children: [
            show list {
                from: realtime.messages
                item as message:
                    // Use the reusable template function
                    message_bubble(message, message.sender_id == session.user_id)
            }
        ]
    }
    
    show div "Team Members" {
        children: [
            show list {
                from: memory.team_members
                layout: "grid",
                columns: 3,
                item as member:
                    show div {
                        style: {
                            padding: "16px",
                            textAlign: "center",
                            border: "1px solid #e5e7eb",
                            borderRadius: "8px"
                        }
                        children: [
                            user_avatar(member, "large"),
                            show text member.name {
                                style: {marginTop: "8px", fontWeight: "600"}
                            },
                            status_badge(member.status, member.status_message)
                        ]
                    }
            }
        ]
    }
}
```

## Embeddable Widget Patterns

### iframe-Safe Embedded Widgets

```namel3ss
// Embeddable chat widget that works in any website
widget "EmbeddableChat" {
    // Widget configuration
    embed_config: {
        width: "400px",
        height: "600px",
        position: "fixed",
        bottom: "20px",
        right: "20px",
        iframe_safe: true,
        cors_origins: ["*"],  // Configure for security
        z_index: 9999
    }
    
    memory: is_minimized = true, unread_count = 0
    realtime: true
    
    // Minimized state - floating button
    if is_minimized:
        show button "Chat Button" {
            style: {
                width: "60px",
                height: "60px",
                borderRadius: "50%",
                backgroundColor: "#3b82f6",
                color: "white",
                border: "none",
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                cursor: "pointer",
                position: "relative"
            }
            
            on_click: {
                update_memory: "is_minimized" = false,
                update_memory: "unread_count" = 0,
                track_event: "chat_widget_opened"
            }
            
            children: [
                show icon "chat",
                if unread_count > 0:
                    show div {
                        style: {
                            position: "absolute",
                            top: "-5px",
                            right: "-5px",
                            backgroundColor: "#dc2626",
                            color: "white",
                            borderRadius: "50%",
                            width: "20px",
                            height: "20px",
                            fontSize: "12px",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                        }
                        children: [show text unread_count]
                    }
            ]
        }
    
    // Expanded state - full chat interface  
    else:
        show div "Chat Widget" {
            style: {
                width: "400px",
                height: "600px",
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 8px 32px rgba(0,0,0,0.12)",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                fontFamily: "system-ui, -apple-system, sans-serif"
            }
            
            children: [
                // Header with minimize button
                show div "Chat Header" {
                    style: {
                        padding: "16px",
                        backgroundColor: "#3b82f6",
                        color: "white",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center"
                    }
                    
                    children: [
                        show text "Customer Support",
                        show button {
                            style: {
                                background: "none",
                                border: "none",
                                color: "white",
                                cursor: "pointer",
                                padding: "4px"
                            }
                            on_click: {
                                update_memory: "is_minimized" = true,
                                track_event: "chat_widget_minimized"
                            }
                            children: [show icon "minimize"]
                        }
                    ]
                },
                
                // Messages area (reuse the message_bubble template)
                show div "Messages" {
                    style: {
                        flex: "1",
                        overflowY: "auto",
                        padding: "16px"
                    }
                    
                    children: [
                        show list {
                            from: realtime.chat_messages,
                            item as message:
                                message_bubble(message, message.sender_id == "user")
                        }
                    ]
                },
                
                // Input area
                show form "Message Form" {
                    style: {
                        padding: "16px",
                        borderTop: "1px solid #e5e7eb",
                        display: "flex",
                        gap: "8px"
                    }
                    
                    fields: [
                        field message: {
                            type: "text",
                            placeholder: "Type your message...",
                            style: {
                                flex: "1",
                                border: "1px solid #e5e7eb",
                                borderRadius: "20px",
                                padding: "8px 16px",
                                outline: "none"
                            }
                        }
                    ]
                    
                    on_submit: {
                        run_chain: "SendChatMessage",
                        clear_form: true,
                        track_event: "message_sent"
                    }
                }
            ]
        }
    
    // Listen for new messages when minimized
    on_realtime_update: {
        if is_minimized && update.type == "new_message":
            update_memory: "unread_count" = unread_count + 1,
            show_notification: {
                title: "New message",
                body: update.message.content,
                icon: "/chat-icon.png"
            }
    }
}
```

### Widget Embedding Script

```namel3ss
// Generate embedding script for websites
script "EmbedChatWidget" {
    output: """
    <script>
    (function() {
        // Create iframe container
        var container = document.createElement('div');
        container.id = 'namel3ss-chat-widget';
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            width: 400px;
            height: 600px;
        `;
        
        // Create iframe
        var iframe = document.createElement('iframe');
        iframe.src = 'https://your-domain.com/widgets/chat';
        iframe.style.cssText = `
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 12px;
        `;
        
        container.appendChild(iframe);
        document.body.appendChild(container);
        
        // Post message communication for events
        window.addEventListener('message', function(event) {
            if (event.origin !== 'https://your-domain.com') return;
            
            if (event.data.type === 'namel3ss-resize') {
                container.style.width = event.data.width + 'px';
                container.style.height = event.data.height + 'px';
            }
        });
    })();
    </script>
    """
}
```

## Advanced Navigation Patterns

### Breadcrumb Navigation

```namel3ss
page "ProductDetails" at "/product/{product_id}" {
    memory: breadcrumbs = [
        {label: "Home", url: "/"},
        {label: "Products", url: "/products"},
        {label: product.category, url: "/products/{product.category}"},
        {label: product.name, url: "/product/{product.id}", current: true}
    ]
    
    show breadcrumb_nav {
        items: breadcrumbs,
        separator: "/",
        style: {
            margin: "16px 0",
            fontSize: "14px",
            color: "#6b7280"
        },
        on_item_click: {
            navigate: item.url,
            track_event: "breadcrumb_navigation"
        }
    }
}
```

### Contextual Navigation

```namel3ss
page "OrderManagement" at "/orders" {
    memory: selected_orders = [], bulk_action = null
    
    show div "Navigation Bar" {
        children: [
            show button_group [
                {
                    label: "All Orders",
                    active: route.params.status == null,
                    on_click: {navigate: "/orders"}
                },
                {
                    label: "Pending", 
                    active: route.params.status == "pending",
                    on_click: {navigate: "/orders?status=pending"}
                },
                {
                    label: "Completed",
                    active: route.params.status == "completed", 
                    on_click: {navigate: "/orders?status=completed"}
                }
            ],
            
            // Contextual bulk actions when orders are selected
            if selected_orders.length > 0:
                show dropdown "Bulk Actions" {
                    options: [
                        {label: "Mark as Shipped", value: "ship"},
                        {label: "Cancel Orders", value: "cancel"},
                        {label: "Export to CSV", value: "export"}
                    ],
                    on_select: {
                        run_chain: "BulkOrderAction",
                        payload: {action: option.value, order_ids: selected_orders},
                        clear_selection: true,
                        show_toast: "Bulk action completed"
                    }
                }
        ]
    }
}
```

This comprehensive guide demonstrates how Namel3ss provides powerful API integration, flexible navigation patterns, reusable template functions, and embeddable widget capabilities that make it perfect for building complex, interactive applications.