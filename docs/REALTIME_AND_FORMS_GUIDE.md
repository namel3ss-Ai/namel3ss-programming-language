# Real-Time Updates & Advanced Forms Guide

This guide covers advanced real-time features, complex form handling, and interactive UI flows in Namel3ss.

## Real-Time WebSocket Features

### Auto-Subscribing Pages

```namel3ss
page "Dashboard" at "/dashboard" {
    realtime: true
    subscribe_to: ["user_notifications", "system_alerts", "order_updates"]
    
    show div {
        children: [
            show list "Live Notifications" {
                from: realtime.user_notifications
                max_items: 5
                auto_scroll: "bottom"
                item as notification:
                    show div {
                        style: {
                            padding: "12px",
                            margin: "8px 0",
                            borderRadius: "8px",
                            backgroundColor: notification.type == "error" ? "#fef2f2" : "#f0f9ff",
                            borderLeft: "4px solid " + (notification.type == "error" ? "#dc2626" : "#2563eb")
                        }
                        children: [
                            show text notification.message,
                            show text notification.timestamp {
                                style: {fontSize: "12px", color: "#6b7280"}
                            }
                        ]
                    }
            }
        ]
    }
}
```

### Broadcasting from Chains

```namel3ss
chain "ProcessOrder" {
    input: order_data
    
    step process:
        prompt: "Process this order: {input}"
        
    step update_db:
        run_query: "INSERT INTO orders ..."
        
    step notify_users:
        broadcast: {
            channel: "order_updates",
            message: {
                type: "order_created",
                order_id: process.order_id,
                customer: order_data.customer_name
            }
        }
        
    output: process.result
}
```

### Real-Time Data Synchronization

```namel3ss
page "CollaborativeEditor" at "/editor/{doc_id}" {
    realtime: true
    sync_interval: "500ms"  // High-frequency sync
    
    show textarea "Document Content" {
        value: memory.document_content
        on_change: {
            debounce: "300ms",  // Prevent spam
            update_memory: "document_content" = value,
            broadcast: {
                channel: "document_{doc_id}",
                operation: "text_update",
                data: {
                    content: value,
                    cursor_position: cursor,
                    user_id: session.user_id
                }
            }
        }
        
        on_realtime_update: {
            if update.user_id != session.user_id:
                // Apply remote changes without disrupting local cursor
                merge_content: update.content
        }
        
        style: {
            width: "100%",
            height: "400px",
            fontFamily: "monospace",
            fontSize: "14px",
            padding: "16px",
            border: "1px solid #e5e7eb",
            borderRadius: "8px"
        }
    }
    
    show div "Active Users" {
        style: {marginTop: "16px"}
        children: [
            show text "Currently editing:",
            show list {
                from: realtime.active_users
                layout: "horizontal"
                item as user:
                    show div {
                        style: {
                            display: "inline-block",
                            padding: "4px 8px",
                            margin: "0 4px",
                            backgroundColor: user.cursor_color,
                            color: "white",
                            borderRadius: "12px",
                            fontSize: "12px"
                        }
                        children: [show text user.name]
                    }
            }
        ]
    }
}
```

## Advanced Form Features

### Multi-Step Form Workflows

```namel3ss
page "Onboarding" at "/onboarding" {
    memory: form_step = 1, user_data = {}
    
    show div "Progress Bar" {
        style: {
            width: "100%",
            height: "4px",
            backgroundColor: "#e5e7eb",
            borderRadius: "2px",
            marginBottom: "24px",
            overflow: "hidden"
        }
        children: [
            show div {
                style: {
                    width: "{(form_step / 3) * 100}%",
                    height: "100%",
                    backgroundColor: "#3b82f6",
                    transition: "width 0.3s ease"
                }
            }
        ]
    }
    
    if form_step == 1:
        show form "Personal Information" {
            fields: [
                field first_name: {type: "text", required: true},
                field last_name: {type: "text", required: true},
                field email: {type: "email", required: true}
            ]
            
            on_submit: {
                update_memory: "user_data.personal" = form,
                update_memory: "form_step" = 2,
                refresh: true
            }
            
            submit_button: "Next Step"
        }
    
    elif form_step == 2:
        show form "Company Details" {
            fields: [
                field company_name: {type: "text", required: true},
                field role: {
                    type: "select",
                    options: ["Developer", "Designer", "Manager", "Other"],
                    required: true
                },
                field team_size: {
                    type: "select",
                    options: ["1-5", "6-20", "21-50", "50+"],
                    required: true
                }
            ]
            
            on_submit: {
                update_memory: "user_data.company" = form,
                update_memory: "form_step" = 3,
                refresh: true
            }
            
            submit_button: "Almost Done"
        }
        
        show button "Back" {
            on_click: {
                update_memory: "form_step" = 1,
                refresh: true
            }
            style: {
                backgroundColor: "transparent",
                color: "#6b7280",
                border: "1px solid #e5e7eb"
            }
        }
    
    elif form_step == 3:
        show form "Preferences" {
            fields: [
                field newsletter: {
                    type: "checkbox",
                    label: "Subscribe to our newsletter"
                },
                field notifications: {
                    type: "checkbox",
                    label: "Enable email notifications"
                }
            ]
            
            on_submit: {
                update_memory: "user_data.preferences" = form,
                run_chain: "CompleteOnboarding",
                redirect: "/dashboard"
            }
            
            submit_button: "Complete Setup"
        }
        
        show button "Back" {
            on_click: {
                update_memory: "form_step" = 2,
                refresh: true
            }
        }
}
```

### Dynamic Form Fields

```namel3ss
page "ProductOrder" at "/order" {
    memory: order_type = "basic", show_shipping = false
    
    show form "Order Form" {
        fields: [
            field product: {
                type: "select",
                options: ["Basic", "Premium", "Enterprise"],
                on_change: {
                    update_memory: "order_type" = value.toLowerCase(),
                    update_memory: "show_shipping" = (value == "Premium" || value == "Enterprise"),
                    refresh: true
                }
            },
            
            // Conditional fields based on selection
            if order_type == "premium":
                field support_level: {
                    type: "select",
                    options: ["Standard", "Priority", "Dedicated"],
                    required: true
                }
            
            if order_type == "enterprise":
                field custom_features: {
                    type: "textarea",
                    placeholder: "Describe your custom requirements"
                }
                
            if show_shipping:
                field shipping_address: {
                    type: "textarea", 
                    required: true,
                    label: "Shipping Address"
                }
                field express_shipping: {
                    type: "checkbox",
                    label: "Express shipping (+$20)"
                }
        ]
        
        on_submit: {
            validate: {
                if order_type == "enterprise" && !custom_features:
                    show_error: "Custom features required for Enterprise"
            },
            run_chain: "ProcessOrder",
            show_toast: "Order submitted successfully!",
            redirect: "/order-confirmation"
        }
    }
}
```

### Form Validation with Real-Time Feedback

```namel3ss
page "Registration" at "/register" {
    memory: username_available = null, email_valid = null
    
    show form "Create Account" {
        validation_mode: "on_change"  // Real-time validation
        
        fields: [
            field username: {
                type: "text",
                required: true,
                debounce: "500ms",
                on_change: {
                    run_chain: "CheckUsernameAvailability",
                    update_memory: "username_available" = result.available
                },
                validation: {
                    min_length: 3,
                    pattern: "[a-zA-Z0-9_]+",
                    custom: username_available == false ? "Username already taken" : null
                },
                helper_text: username_available == true ? "✓ Username available" : null
            },
            
            field email: {
                type: "email",
                required: true,
                on_change: {
                    run_chain: "ValidateEmail",
                    update_memory: "email_valid" = result.valid
                },
                validation: {
                    custom: email_valid == false ? "Invalid email domain" : null
                }
            },
            
            field password: {
                type: "password",
                required: true,
                validation: {
                    min_length: 8,
                    pattern: "(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)",
                    message: "Password must contain uppercase, lowercase, and number"
                }
            },
            
            field confirm_password: {
                type: "password",
                required: true,
                validation: {
                    custom: value != password ? "Passwords don't match" : null
                }
            }
        ]
        
        on_submit: {
            run_chain: "CreateUserAccount",
            show_toast: "Account created! Please check your email.",
            redirect: "/verify-email"
        }
        
        style: {
            maxWidth: "400px",
            margin: "0 auto",
            padding: "24px"
        }
    }
}
```

## Interactive UI Flows

### Conditional Form Sections

```namel3ss
page "SupportTicket" at "/support" {
    memory: issue_type = null, priority = "low"
    
    show form "Submit Ticket" {
        fields: [
            field issue_type: {
                type: "select",
                options: ["Bug Report", "Feature Request", "Technical Support", "Billing"],
                required: true,
                on_change: {
                    update_memory: "issue_type" = value,
                    refresh: true
                }
            }
        ]
    }
    
    // Show different forms based on issue type
    if issue_type == "Bug Report":
        show form "Bug Details" {
            fields: [
                field steps_to_reproduce: {type: "textarea", required: true},
                field browser: {type: "select", options: ["Chrome", "Firefox", "Safari", "Edge"]},
                field screenshot: {type: "file", accept: "image/*"}
            ]
            on_submit: {
                run_chain: "CreateBugTicket",
                show_toast: "Bug report submitted"
            }
        }
    
    elif issue_type == "Feature Request":
        show form "Feature Details" {
            fields: [
                field description: {type: "textarea", required: true},
                field use_case: {type: "textarea"},
                field priority: {
                    type: "select",
                    options: ["Low", "Medium", "High"],
                    on_change: {update_memory: "priority" = value.toLowerCase()}
                }
            ]
            on_submit: {
                run_chain: "CreateFeatureRequest",
                show_toast: "Feature request submitted"
            }
        }
    
    elif issue_type == "Technical Support":
        show form "Technical Issue" {
            fields: [
                field problem_description: {type: "textarea", required: true},
                field urgency: {
                    type: "select",
                    options: ["Low", "Medium", "High", "Critical"],
                    required: true
                },
                field error_logs: {type: "file", accept: ".txt,.log"}
            ]
            on_submit: {
                run_chain: "CreateSupportTicket",
                show_toast: "Support ticket created"
            }
        }
}
```

### Live Search and Filtering

```namel3ss
page "CustomerSearch" at "/customers" {
    memory: search_query = "", search_results = [], loading = false
    
    show div "Search Interface" {
        children: [
            show input "Search Customers" {
                value: search_query,
                placeholder: "Type to search customers...",
                debounce: "300ms",
                on_change: {
                    update_memory: "search_query" = value,
                    update_memory: "loading" = true,
                    run_chain: "SearchCustomers",
                    update_memory: "search_results" = result.customers,
                    update_memory: "loading" = false
                }
                style: {
                    width: "100%",
                    padding: "12px 16px",
                    fontSize: "16px",
                    border: "2px solid #e5e7eb",
                    borderRadius: "8px",
                    outline: "none",
                    transition: "border-color 0.2s"
                }
            },
            
            if loading:
                show div "Loading..." {
                    style: {
                        padding: "16px",
                        textAlign: "center",
                        color: "#6b7280"
                    }
                }
            
            elif search_query && search_results.length > 0:
                show list "Search Results" {
                    from: search_results
                    max_height: "400px"
                    item as customer:
                        show div {
                            style: {
                                padding: "12px 16px",
                                borderBottom: "1px solid #e5e7eb",
                                cursor: "pointer",
                                hover: {backgroundColor: "#f9fafb"}
                            }
                            on_click: {
                                navigate: "/customer/{customer.id}"
                            }
                            children: [
                                show text customer.name {
                                    style: {fontWeight: "600", fontSize: "16px"}
                                },
                                show text customer.email {
                                    style: {color: "#6b7280", fontSize: "14px"}
                                },
                                show text "Customer since {customer.created_date}" {
                                    style: {color: "#9ca3af", fontSize: "12px"}
                                }
                            ]
                        }
                }
            
            elif search_query && search_results.length == 0:
                show div "No customers found" {
                    style: {
                        padding: "32px",
                        textAlign: "center",
                        color: "#6b7280"
                    }
                }
        ]
    }
}
```

## Real-Time Chat Implementation

### Complete Chat Widget

```namel3ss
page "ChatWidget" at "/chat" {
    realtime: true
    subscribe_to: ["chat_messages", "user_presence"]
    memory: message_text = "", is_typing = false
    
    show div "Chat Container" {
        style: {
            height: "500px",
            display: "flex",
            flexDirection: "column",
            border: "1px solid #e5e7eb",
            borderRadius: "12px",
            backgroundColor: "white"
        }
        
        children: [
            // Chat header
            show div "Chat Header" {
                style: {
                    padding: "16px",
                    borderBottom: "1px solid #e5e7eb",
                    backgroundColor: "#f8fafc"
                }
                children: [
                    show text "Customer Support",
                    show div {
                        style: {fontSize: "12px", color: "#6b7280"}
                        children: [
                            show text "Online agents: {realtime.online_agents.length}"
                        ]
                    }
                ]
            },
            
            // Messages area
            show div "Messages" {
                style: {
                    flex: "1",
                    overflow: "auto",
                    padding: "16px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px"
                }
                
                children: [
                    show list {
                        from: realtime.chat_messages
                        auto_scroll: "bottom"
                        item as message:
                            show div {
                                style: {
                                    display: "flex",
                                    justifyContent: message.is_agent ? "flex-start" : "flex-end"
                                }
                                children: [
                                    show div {
                                        style: {
                                            maxWidth: "70%",
                                            padding: "8px 12px",
                                            borderRadius: "12px",
                                            backgroundColor: message.is_agent ? "#f3f4f6" : "#3b82f6",
                                            color: message.is_agent ? "#111827" : "white"
                                        }
                                        children: [
                                            show text message.content,
                                            show text message.timestamp {
                                                style: {
                                                    fontSize: "10px",
                                                    opacity: "0.7",
                                                    marginTop: "4px"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                    }
                ]
            },
            
            // Typing indicator
            if realtime.agent_typing:
                show div "Agent is typing..." {
                    style: {
                        padding: "8px 16px",
                        fontSize: "12px",
                        color: "#6b7280",
                        fontStyle: "italic"
                    }
                }
            
            // Message input
            show form "Send Message" {
                style: {
                    padding: "16px",
                    borderTop: "1px solid #e5e7eb",
                    display: "flex",
                    gap: "8px"
                }
                
                fields: [
                    field message_text: {
                        type: "text",
                        placeholder: "Type your message...",
                        on_change: {
                            update_memory: "message_text" = value,
                            if !is_typing:
                                update_memory: "is_typing" = true,
                                broadcast: {
                                    channel: "chat_typing",
                                    data: {user_id: session.user_id, typing: true}
                                }
                        },
                        on_blur: {
                            update_memory: "is_typing" = false,
                            broadcast: {
                                channel: "chat_typing", 
                                data: {user_id: session.user_id, typing: false}
                            }
                        }
                    }
                ]
                
                on_submit: {
                    run_chain: "SendChatMessage",
                    clear_form: true,
                    update_memory: "message_text" = "",
                    broadcast: {
                        channel: "chat_messages",
                        data: {
                            content: message_text,
                            user_id: session.user_id,
                            timestamp: now()
                        }
                    }
                }
                
                submit_button: "Send"
            }
        ]
    }
}
```

This comprehensive guide shows how Namel3ss supports advanced real-time features, complex form workflows, and interactive UI patterns that make it perfect for building sophisticated applications like chat widgets, dashboards, and collaborative tools.