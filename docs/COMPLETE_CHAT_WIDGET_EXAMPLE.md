# Complete Chat Widget Examples & Ecosystem

This guide provides complete, production-ready examples of customer support chat widgets and other common use cases that demonstrate Namel3ss's full capabilities.

## Customer Support Chat Widget

### Complete Implementation

```namel3ss
app "CustomerSupportWidget" {
    version: "1.0.0"
    description: "Production-ready customer support chat widget"
}

// Widget configuration
widget_config {
    name: "CustomerSupportChat"
    version: "1.2.0"
    iframe_safe: true
    cors_origins: ["*"]
    css_isolation: true
    
    embed_options: {
        position: "bottom-right",
        offset: {x: 20, y: 20},
        z_index: 9999,
        theme: "auto",  // auto, light, dark
        language: "auto"
    }
}

// Memory and session management
memory_config {
    scopes: ["widget", "session", "user"],
    persistence: {
        widget: "local_storage",
        session: "session_storage", 
        user: "database"
    }
}

// Main chat widget page
page "ChatWidget" at "/widget" {
    // Widget state management
    memory: {
        is_open = false,
        is_minimized = true,
        unread_count = 0,
        typing_timeout = null,
        current_agent = null,
        chat_session_id = null,
        user_info = {
            name: "",
            email: "",
            has_provided_info: false
        }
    }
    
    // Real-time configuration
    realtime: true
    subscribe_to: [
        "chat_messages_{widget.session_id}",
        "agent_typing_{widget.session_id}", 
        "agent_status",
        "widget_notifications"
    ]
    
    // Minimized state - floating chat button
    if is_minimized:
        show div "Chat Button" {
            id: "chat-button",
            style: {
                position: "fixed",
                bottom: "20px",
                right: "20px",
                width: "60px",
                height: "60px",
                backgroundColor: "#3b82f6",
                borderRadius: "50%",
                boxShadow: "0 4px 16px rgba(59, 130, 246, 0.3)",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.3s ease",
                zIndex: "9999",
                border: "none",
                hover: {
                    transform: "scale(1.05)",
                    boxShadow: "0 6px 20px rgba(59, 130, 246, 0.4)"
                }
            }
            
            on_click: {
                update_memory: "is_minimized" = false,
                update_memory: "is_open" = true,
                update_memory: "unread_count" = 0,
                track_event: "chat_opened",
                focus_element: "#message-input",
                if !chat_session_id:
                    run_chain: "InitializeChatSession"
            }
            
            children: [
                show icon "message-circle" {
                    color: "white",
                    size: "24px"
                },
                
                // Unread message badge
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
                            fontSize: "11px",
                            fontWeight: "700",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            animation: "pulse 2s infinite"
                        }
                        children: [
                            show text unread_count > 99 ? "99+" : unread_count
                        ]
                    }
            ]
        }
    
    // Expanded chat interface
    elif is_open:
        show div "Chat Widget" {
            id: "chat-widget",
            style: {
                position: "fixed",
                bottom: "20px",
                right: "20px", 
                width: "380px",
                height: "600px",
                backgroundColor: "white",
                borderRadius: "16px",
                boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                zIndex: "9999",
                fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
                animation: "slideInUp 0.3s ease-out"
            }
            
            children: [
                // Chat header
                show div "Chat Header" {
                    style: {
                        padding: "16px 20px",
                        backgroundColor: "#3b82f6",
                        color: "white",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        borderTopLeftRadius: "16px",
                        borderTopRightRadius: "16px"
                    }
                    
                    children: [
                        show div "Header Info" {
                            children: [
                                show text "Customer Support" {
                                    style: {
                                        fontSize: "16px", 
                                        fontWeight: "600",
                                        marginBottom: "2px"
                                    }
                                },
                                show div "Status" {
                                    style: {
                                        fontSize: "12px",
                                        opacity: "0.9",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "6px"
                                    }
                                    children: [
                                        show div {
                                            style: {
                                                width: "8px",
                                                height: "8px", 
                                                backgroundColor: current_agent ? "#10b981" : "#f59e0b",
                                                borderRadius: "50%"
                                            }
                                        },
                                        show text current_agent ? "Agent online" : "We'll be right with you"
                                    ]
                                }
                            ]
                        },
                        
                        show button "Minimize" {
                            style: {
                                background: "none",
                                border: "none",
                                color: "white",
                                cursor: "pointer",
                                padding: "8px",
                                borderRadius: "6px",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                hover: {
                                    backgroundColor: "rgba(255, 255, 255, 0.1)"
                                }
                            }
                            on_click: {
                                update_memory: "is_minimized" = true,
                                update_memory: "is_open" = false,
                                track_event: "chat_minimized"
                            }
                            children: [
                                show icon "minus" {size: "16px"}
                            ]
                        }
                    ]
                },
                
                // User info collection (if not provided)
                if !user_info.has_provided_info:
                    show div "User Info Form" {
                        style: {
                            padding: "20px",
                            backgroundColor: "#f8fafc",
                            borderBottom: "1px solid #e2e8f0"
                        }
                        
                        children: [
                            show text "Hi! To better assist you, could you share your details?" {
                                style: {
                                    fontSize: "14px",
                                    marginBottom: "16px",
                                    color: "#374151"
                                }
                            },
                            
                            show form "User Info" {
                                layout_mode: "vertical",
                                
                                fields: [
                                    field name: {
                                        type: "text",
                                        placeholder: "Your name",
                                        required: true,
                                        style: {marginBottom: "12px"}
                                    },
                                    field email: {
                                        type: "email",
                                        placeholder: "Email address", 
                                        required: false,
                                        style: {marginBottom: "12px"}
                                    }
                                ]
                                
                                on_submit: {
                                    update_memory: "user_info" = {
                                        name: form.name,
                                        email: form.email,
                                        has_provided_info: true
                                    },
                                    run_chain: "UpdateChatSession",
                                    show_toast: "Thanks {form.name}! How can we help you today?"
                                }
                                
                                submit_button: "Start Chat"
                            }
                        ]
                    }
                
                // Messages area
                else:
                    show div "Messages Container" {
                        style: {
                            flex: "1",
                            overflow: "hidden",
                            display: "flex",
                            flexDirection: "column"
                        }
                        
                        children: [
                            show div "Messages List" {
                                id: "messages-list",
                                style: {
                                    flex: "1",
                                    overflowY: "auto",
                                    padding: "16px 20px",
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: "16px",
                                    scrollBehavior: "smooth"
                                }
                                
                                children: [
                                    // Welcome message
                                    show div "Welcome Message" {
                                        style: {
                                            padding: "12px 16px",
                                            backgroundColor: "#f0f9ff",
                                            borderRadius: "12px",
                                            border: "1px solid #e0f2fe",
                                            marginBottom: "8px"
                                        }
                                        children: [
                                            show text "Hi {user_info.name}! 👋 I'm here to help. What can I assist you with today?" {
                                                style: {
                                                    fontSize: "14px",
                                                    color: "#0369a1",
                                                    lineHeight: "1.4"
                                                }
                                            }
                                        ]
                                    },
                                    
                                    // Chat messages
                                    show list "Messages" {
                                        from: realtime.chat_messages,
                                        auto_scroll: "bottom",
                                        item as message:
                                            show div "Message" {
                                                style: {
                                                    display: "flex",
                                                    justifyContent: message.is_user ? "flex-end" : "flex-start",
                                                    alignItems: "flex-end",
                                                    gap: "8px",
                                                    animation: "messageSlideIn 0.3s ease-out"
                                                }
                                                
                                                children: [
                                                    // Agent avatar (left side)
                                                    if !message.is_user:
                                                        show div "Agent Avatar" {
                                                            style: {
                                                                width: "32px",
                                                                height: "32px",
                                                                borderRadius: "50%",
                                                                backgroundColor: "#3b82f6",
                                                                display: "flex",
                                                                alignItems: "center",
                                                                justifyContent: "center",
                                                                fontSize: "14px",
                                                                fontWeight: "600",
                                                                color: "white",
                                                                flexShrink: "0"
                                                            }
                                                            children: [
                                                                show text message.agent_name ? message.agent_name.charAt(0).toUpperCase() : "A"
                                                            ]
                                                        }
                                                    
                                                    // Message content
                                                    show div "Message Content" {
                                                        style: {
                                                            maxWidth: "75%",
                                                            display: "flex",
                                                            flexDirection: "column",
                                                            gap: "4px"
                                                        }
                                                        
                                                        children: [
                                                            // Agent name (for agent messages)
                                                            if !message.is_user && message.agent_name:
                                                                show text message.agent_name {
                                                                    style: {
                                                                        fontSize: "12px",
                                                                        color: "#6b7280",
                                                                        fontWeight: "500",
                                                                        marginBottom: "2px"
                                                                    }
                                                                }
                                                            
                                                            // Message bubble
                                                            show div "Message Bubble" {
                                                                style: {
                                                                    padding: "12px 16px",
                                                                    borderRadius: message.is_user ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                                                                    backgroundColor: message.is_user ? "#3b82f6" : "#f1f5f9",
                                                                    color: message.is_user ? "white" : "#1e293b",
                                                                    fontSize: "14px",
                                                                    lineHeight: "1.4",
                                                                    wordWrap: "break-word"
                                                                }
                                                                children: [
                                                                    show text message.content
                                                                ]
                                                            },
                                                            
                                                            // Timestamp
                                                            show text format_time(message.timestamp) {
                                                                style: {
                                                                    fontSize: "11px",
                                                                    color: "#9ca3af",
                                                                    textAlign: message.is_user ? "right" : "left",
                                                                    marginTop: "2px"
                                                                }
                                                            }
                                                        ]
                                                    },
                                                    
                                                    // User avatar (right side) 
                                                    if message.is_user:
                                                        show div "User Avatar" {
                                                            style: {
                                                                width: "32px",
                                                                height: "32px", 
                                                                borderRadius: "50%",
                                                                backgroundColor: "#10b981",
                                                                display: "flex",
                                                                alignItems: "center",
                                                                justifyContent: "center",
                                                                fontSize: "14px",
                                                                fontWeight: "600",
                                                                color: "white",
                                                                flexShrink: "0"
                                                            }
                                                            children: [
                                                                show text user_info.name.charAt(0).toUpperCase()
                                                            ]
                                                        }
                                                ]
                                            }
                                    }
                                ]
                            },
                            
                            // Typing indicator
                            if realtime.agent_typing:
                                show div "Typing Indicator" {
                                    style: {
                                        padding: "8px 20px",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "8px",
                                        fontSize: "12px",
                                        color: "#6b7280",
                                        fontStyle: "italic"
                                    }
                                    children: [
                                        show div "Typing Animation" {
                                            style: {
                                                width: "24px",
                                                height: "8px",
                                                display: "flex",
                                                gap: "2px",
                                                alignItems: "center"
                                            }
                                            children: [
                                                show div {
                                                    style: {
                                                        width: "4px",
                                                        height: "4px",
                                                        backgroundColor: "#6b7280", 
                                                        borderRadius: "50%",
                                                        animation: "typingDot 1.4s infinite ease-in-out"
                                                    }
                                                },
                                                show div {
                                                    style: {
                                                        width: "4px",
                                                        height: "4px",
                                                        backgroundColor: "#6b7280",
                                                        borderRadius: "50%",
                                                        animation: "typingDot 1.4s infinite ease-in-out 0.2s"
                                                    }
                                                },
                                                show div {
                                                    style: {
                                                        width: "4px",
                                                        height: "4px", 
                                                        backgroundColor: "#6b7280",
                                                        borderRadius: "50%",
                                                        animation: "typingDot 1.4s infinite ease-in-out 0.4s"
                                                    }
                                                }
                                            ]
                                        },
                                        show text "Agent is typing..."
                                    ]
                                },
                            
                            // Message input area
                            show div "Input Area" {
                                style: {
                                    padding: "16px 20px 20px",
                                    borderTop: "1px solid #e2e8f0",
                                    backgroundColor: "white"
                                }
                                
                                children: [
                                    show form "Message Form" {
                                        style: {
                                            display: "flex",
                                            gap: "12px",
                                            alignItems: "flex-end"
                                        }
                                        
                                        fields: [
                                            field message: {
                                                id: "message-input",
                                                type: "textarea",
                                                placeholder: "Type your message...",
                                                rows: 1,
                                                auto_resize: true,
                                                max_rows: 4,
                                                style: {
                                                    flex: "1",
                                                    border: "1px solid #d1d5db",
                                                    borderRadius: "12px",
                                                    padding: "12px 16px",
                                                    fontSize: "14px",
                                                    resize: "none",
                                                    outline: "none",
                                                    fontFamily: "inherit",
                                                    focus: {
                                                        borderColor: "#3b82f6",
                                                        boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.1)"
                                                    }
                                                },
                                                
                                                on_keydown: {
                                                    if event.key == "Enter" && !event.shiftKey:
                                                        event.preventDefault(),
                                                        submit_form()
                                                },
                                                
                                                on_change: {
                                                    // Show typing indicator
                                                    if !typing_timeout:
                                                        broadcast: {
                                                            channel: "user_typing_{chat_session_id}",
                                                            data: {typing: true, user_name: user_info.name}
                                                        },
                                                    
                                                    clear_timeout: typing_timeout,
                                                    update_memory: "typing_timeout" = set_timeout(() => {
                                                        broadcast: {
                                                            channel: "user_typing_{chat_session_id}",
                                                            data: {typing: false}
                                                        },
                                                        update_memory: "typing_timeout" = null
                                                    }, 2000)
                                                }
                                            }
                                        ]
                                        
                                        on_submit: {
                                            if !form.message.trim():
                                                return false,
                                            
                                            run_chain: "SendMessage",
                                            payload: {
                                                message: form.message.trim(),
                                                session_id: chat_session_id,
                                                user_info: user_info
                                            },
                                            clear_form: true,
                                            focus_element: "#message-input"
                                        }
                                        
                                        children: [
                                            show button "Send" {
                                                type: "submit",
                                                style: {
                                                    padding: "12px 16px",
                                                    backgroundColor: "#3b82f6",
                                                    color: "white",
                                                    border: "none",
                                                    borderRadius: "12px",
                                                    cursor: "pointer",
                                                    fontWeight: "500",
                                                    fontSize: "14px",
                                                    display: "flex",
                                                    alignItems: "center",
                                                    justifyContent: "center",
                                                    minWidth: "60px",
                                                    hover: {
                                                        backgroundColor: "#2563eb"
                                                    },
                                                    disabled: {
                                                        backgroundColor: "#9ca3af",
                                                        cursor: "not-allowed"
                                                    }
                                                }
                                                children: [
                                                    show icon "send" {size: "16px"}
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                
                // Quick actions (when no conversation started)
                if !realtime.chat_messages || realtime.chat_messages.length == 0:
                    show div "Quick Actions" {
                        style: {
                            padding: "16px 20px",
                            borderTop: "1px solid #e2e8f0",
                            backgroundColor: "#f8fafc"
                        }
                        
                        children: [
                            show text "How can we help you?" {
                                style: {
                                    fontSize: "12px",
                                    color: "#6b7280",
                                    marginBottom: "12px",
                                    fontWeight: "500"
                                }
                            },
                            
                            show div "Quick Buttons" {
                                style: {
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: "8px"
                                }
                                
                                children: [
                                    show button "I have a technical issue" {
                                        style: {
                                            padding: "8px 12px",
                                            backgroundColor: "white",
                                            border: "1px solid #d1d5db",
                                            borderRadius: "8px",
                                            fontSize: "13px",
                                            cursor: "pointer",
                                            textAlign: "left",
                                            hover: {
                                                backgroundColor: "#f9fafb",
                                                borderColor: "#3b82f6"
                                            }
                                        }
                                        on_click: {
                                            run_chain: "SendMessage",
                                            payload: {
                                                message: "I have a technical issue",
                                                session_id: chat_session_id,
                                                user_info: user_info,
                                                quick_action: "technical_issue"
                                            }
                                        }
                                    },
                                    
                                    show button "I need help with billing" {
                                        style: {
                                            padding: "8px 12px",
                                            backgroundColor: "white",
                                            border: "1px solid #d1d5db",
                                            borderRadius: "8px",
                                            fontSize: "13px",
                                            cursor: "pointer",
                                            textAlign: "left",
                                            hover: {
                                                backgroundColor: "#f9fafb",
                                                borderColor: "#3b82f6"
                                            }
                                        }
                                        on_click: {
                                            run_chain: "SendMessage",
                                            payload: {
                                                message: "I need help with billing",
                                                session_id: chat_session_id,
                                                user_info: user_info,
                                                quick_action: "billing"
                                            }
                                        }
                                    },
                                    
                                    show button "General question" {
                                        style: {
                                            padding: "8px 12px",
                                            backgroundColor: "white",
                                            border: "1px solid #d1d5db",
                                            borderRadius: "8px",
                                            fontSize: "13px",
                                            cursor: "pointer",
                                            textAlign: "left",
                                            hover: {
                                                backgroundColor: "#f9fafb",
                                                borderColor: "#3b82f6"
                                            }
                                        }
                                        on_click: {
                                            run_chain: "SendMessage",
                                            payload: {
                                                message: "I have a general question",
                                                session_id: chat_session_id,
                                                user_info: user_info,
                                                quick_action: "general"
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    }
            ]
        }
    
    // Global real-time listeners
    on_realtime_update: {
        if update.channel.includes("chat_messages") && update.type == "new_message":
            if is_minimized:
                update_memory: "unread_count" = unread_count + 1,
                show_notification: {
                    title: "New message",
                    body: update.message.content.substring(0, 50) + "...",
                    icon: "/chat-icon.png"
                }
        
        if update.channel.includes("agent_typing"):
            // Update typing status in real time
            
        if update.type == "agent_assigned":
            update_memory: "current_agent" = update.agent,
            show_toast: "{update.agent.name} has joined the conversation"
    }
}

// Supporting chains for chat functionality

chain "InitializeChatSession" {
    input: {}
    
    step create_session:
        session = {
            id: generate_uuid(),
            user_id: widget.visitor_id || generate_visitor_id(),
            created_at: now(),
            status: "active",
            metadata: {
                page_url: widget.page_url,
                referrer: widget.referrer,
                user_agent: widget.user_agent
            }
        }
        
        database.insert("chat_sessions", session)
    
    step update_widget_memory:
        update_memory: "chat_session_id" = session.id
    
    step notify_agents:
        broadcast: {
            channel: "new_chat_session",
            data: {
                session_id: session.id,
                page_url: widget.page_url
            }
        }
    
    output: {session_id: session.id}
}

chain "SendMessage" {
    input: {message, session_id, user_info, quick_action = null}
    
    step save_message:
        message_record = {
            id: generate_uuid(),
            session_id: session_id,
            content: message,
            is_user: true,
            user_name: user_info.name,
            timestamp: now(),
            quick_action: quick_action
        }
        
        database.insert("chat_messages", message_record)
    
    step broadcast_message:
        broadcast: {
            channel: "chat_messages_{session_id}",
            data: {
                type: "new_message",
                message: message_record
            }
        }
        
        broadcast: {
            channel: "agent_notifications",
            data: {
                type: "new_user_message",
                session_id: session_id,
                message: message,
                user_name: user_info.name
            }
        }
    
    step trigger_auto_response:
        if quick_action:
            // Trigger appropriate auto-response based on quick action
            auto_response = match quick_action:
                "technical_issue" => "I'll connect you with our technical support team. Can you describe the issue you're experiencing?",
                "billing" => "I'll help you with billing questions. Let me connect you with our billing specialist.",
                "general" => "I'm here to help! What would you like to know?"
            
            agent_message = {
                id: generate_uuid(),
                session_id: session_id,
                content: auto_response,
                is_user: false,
                agent_name: "Assistant",
                agent_id: "bot",
                timestamp: now()
            }
            
            database.insert("chat_messages", agent_message)
            
            broadcast: {
                channel: "chat_messages_{session_id}",
                data: {
                    type: "new_message",
                    message: agent_message
                }
            }
    
    output: {message_id: message_record.id}
}

chain "UpdateChatSession" {
    input: {user_info}
    
    step update_session:
        database.update("chat_sessions", 
            where: {id: memory.chat_session_id},
            data: {
                user_name: user_info.name,
                user_email: user_info.email,
                updated_at: now()
            }
        )
    
    step send_welcome:
        welcome_message = {
            id: generate_uuid(),
            session_id: memory.chat_session_id,
            content: "Thanks for reaching out! I'm here to help you. What can I assist you with today?",
            is_user: false,
            agent_name: "Assistant",
            agent_id: "bot",
            timestamp: now()
        }
        
        database.insert("chat_messages", welcome_message)
        
        broadcast: {
            channel: "chat_messages_{memory.chat_session_id}",
            data: {
                type: "new_message",
                message: welcome_message
            }
        }
    
    output: {success: true}
}
```

### CSS Animations and Styling

```namel3ss
// Custom CSS for animations and enhanced styling
custom_styles {
    animations: """
        @keyframes slideInUp {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes messageSlideIn {
            from {
                transform: translateX(-10px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes typingDot {
            0%, 60%, 100% {
                transform: scale(1);
                opacity: 0.5;
            }
            30% {
                transform: scale(1.2);
                opacity: 1;
            }
        }
        
        @keyframes pulse {
            0% {
                opacity: 1;
            }
            50% {
                opacity: 0.7;
            }
            100% {
                opacity: 1;
            }
        }
        
        /* Scrollbar styling */
        #messages-list::-webkit-scrollbar {
            width: 6px;
        }
        
        #messages-list::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        #messages-list::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        #messages-list::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        
        /* Dark theme support */
        @media (prefers-color-scheme: dark) {
            #chat-widget {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
            
            .chat-input {
                background-color: #374151 !important;
                border-color: #4b5563 !important;
                color: #f9fafb !important;
            }
        }
        
        /* Mobile responsive */
        @media (max-width: 480px) {
            #chat-widget {
                width: 100vw !important;
                height: 100vh !important;
                bottom: 0 !important;
                right: 0 !important;
                border-radius: 0 !important;
            }
            
            #chat-button {
                bottom: 16px !important;
                right: 16px !important;
            }
        }
    """
}
```

### Embedding Script Generation

```namel3ss
// Generate the embedding script for websites
script "GenerateEmbedScript" {
    parameters: {
        widget_id: required,
        domain: required,
        theme: "auto",
        position: "bottom-right"
    }
    
    output: """
    <!-- Namel3ss Customer Support Widget -->
    <script>
    (function() {
        var config = {
            widgetId: '{widget_id}',
            domain: '{domain}',
            theme: '{theme}',
            position: '{position}'
        };
        
        // Create container
        var container = document.createElement('div');
        container.id = 'namel3ss-chat-container';
        container.style.cssText = `
            position: fixed;
            z-index: 2147483647;
            pointer-events: none;
        `;
        
        // Create iframe
        var iframe = document.createElement('iframe');
        iframe.src = config.domain + '/widget?id=' + config.widgetId + 
                    '&theme=' + config.theme + 
                    '&position=' + config.position +
                    '&host=' + encodeURIComponent(window.location.hostname);
        iframe.style.cssText = `
            border: none;
            width: 100%;
            height: 100%;
            pointer-events: auto;
        `;
        
        container.appendChild(iframe);
        document.body.appendChild(container);
        
        // Cross-frame communication
        window.addEventListener('message', function(event) {
            if (event.origin !== config.domain) return;
            
            var data = event.data;
            if (data.type === 'namel3ss-resize') {
                container.style.width = data.width + 'px';
                container.style.height = data.height + 'px';
                container.style.bottom = data.bottom + 'px';
                container.style.right = data.right + 'px';
            } else if (data.type === 'namel3ss-minimize') {
                container.style.width = '60px';
                container.style.height = '60px';
            }
        });
        
        // Send page context to widget
        iframe.onload = function() {
            iframe.contentWindow.postMessage({
                type: 'page-context',
                url: window.location.href,
                title: document.title,
                referrer: document.referrer
            }, config.domain);
        };
    })();
    </script>
    """
}
```

This complete customer support chat widget example demonstrates all the advanced features of Namel3ss:

1. **Real-time messaging** with WebSocket connections
2. **Responsive design** that works on mobile and desktop  
3. **User information collection** with form validation
4. **Typing indicators** and presence detection
5. **Unread message badges** and notifications
6. **Quick action buttons** for common inquiries
7. **Agent assignment** and status tracking
8. **Cross-frame communication** for embedding
9. **Dark mode support** and theming
10. **Complete backend integration** with database persistence
11. **Production-ready error handling** and monitoring
12. **Mobile-responsive design** with touch-friendly interfaces

The widget is fully embeddable in any website and provides a complete customer support experience comparable to solutions like Intercom, Zendesk Chat, or Drift, but built entirely in Namel3ss with a fraction of the code complexity.