# Backend Features & Error Handling Guide

This guide covers advanced backend features, session management, data persistence, error handling, and production deployment patterns in Namel3ss.

## Advanced Backend Features

### Custom Backend Logic

```namel3ss
// Define custom backend functions
backend_function "CalculateShipping" {
    inputs: {address, items, shipping_method}
    
    logic: {
        // Complex business logic
        base_rate = shipping_method == "express" ? 15.00 : 5.00
        weight = items.reduce((sum, item) => sum + item.weight, 0)
        distance_rate = calculate_distance(address.zip_code, warehouse.zip_code) * 0.10
        
        total = base_rate + (weight * 0.50) + distance_rate
        
        // Apply discounts
        if total > 50:
            total = total * 0.9  // 10% discount for orders over $50
        
        return {
            cost: total.toFixed(2),
            estimated_days: shipping_method == "express" ? 1 : 3,
            provider: shipping_method == "express" ? "FedEx" : "USPS"
        }
    }
}

// Use in chains
chain "ProcessCheckout" {
    input: checkout_data
    
    step validate_items:
        items = checkout_data.cart_items
        for item in items:
            if item.quantity > inventory.get(item.id):
                throw "Insufficient inventory for {item.name}"
    
    step calculate_shipping:
        shipping = backend_function("CalculateShipping", {
            address: checkout_data.shipping_address,
            items: items,
            shipping_method: checkout_data.shipping_method
        })
    
    step process_payment:
        payment_result = payment_gateway.charge({
            amount: checkout_data.total + shipping.cost,
            card_token: checkout_data.payment_token
        })
        
        if !payment_result.success:
            throw "Payment failed: {payment_result.error}"
    
    step create_order:
        order = database.insert("orders", {
            user_id: session.user_id,
            items: items,
            shipping_cost: shipping.cost,
            total: checkout_data.total + shipping.cost,
            status: "processing",
            created_at: now()
        })
    
    step send_confirmation:
        email.send({
            to: session.user.email,
            template: "order_confirmation",
            data: {order: order, shipping: shipping}
        })
    
    output: {order_id: order.id, shipping: shipping}
}
```

### Database Operations

```namel3ss
// Advanced database patterns
query "GetUserDashboard" {
    sql: """
        SELECT 
            u.id, u.name, u.email, u.avatar_url,
            COUNT(DISTINCT o.id) as total_orders,
            SUM(o.total) as total_spent,
            MAX(o.created_at) as last_order_date,
            COUNT(DISTINCT CASE WHEN o.created_at > NOW() - INTERVAL '30 days' THEN o.id END) as orders_last_30_days
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.id = :user_id
        GROUP BY u.id, u.name, u.email, u.avatar_url
    """
    
    parameters: {user_id: session.user_id}
    cache_ttl: "5 minutes"
}

query "SearchProducts" {
    sql: """
        SELECT p.*, c.name as category_name,
               ts_rank_cd(search_vector, plainto_tsquery(:search_term)) as relevance_score
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE (:category_id IS NULL OR p.category_id = :category_id)
          AND (:search_term IS NULL OR search_vector @@ plainto_tsquery(:search_term))
          AND p.is_active = true
        ORDER BY 
          CASE WHEN :search_term IS NOT NULL THEN relevance_score END DESC,
          p.popularity_score DESC,
          p.name ASC
        LIMIT :limit OFFSET :offset
    """
    
    parameters: {
        search_term: null,
        category_id: null, 
        limit: 20,
        offset: 0
    }
}

// Use with complex filtering
page "ProductSearch" at "/search" {
    memory: filters = {}, search_results = []
    
    show form "Search Filters" {
        fields: [
            field search_term: {placeholder: "Search products..."},
            field category: {type: "select", options: memory.categories},
            field price_min: {type: "number", placeholder: "Min price"},
            field price_max: {type: "number", placeholder: "Max price"}
        ]
        
        on_submit: {
            run_query: "SearchProducts",
            payload: {
                search_term: form.search_term || null,
                category_id: form.category || null,
                price_min: form.price_min || null,
                price_max: form.price_max || null,
                limit: 20,
                offset: 0
            },
            update_memory: "search_results" = result,
            update_memory: "filters" = form
        }
    }
}
```

## Session Management

### Advanced Session Handling

```namel3ss
// Custom session configuration
session_config {
    provider: "redis",  // redis, memory, database
    ttl: "24 hours",
    secure: true,
    same_site: "lax",
    cookie_name: "namel3ss_session",
    
    // Custom session data
    defaults: {
        preferences: {
            theme: "light",
            language: "en",
            notifications: true
        },
        cart: {
            items: [],
            total: 0
        }
    }
}

// Session-aware components
page "UserPreferences" at "/preferences" {
    // Pre-load user preferences from session
    init: {
        load_from_session: ["preferences", "recent_activity"]
    }
    
    show form "Preferences" {
        initial_values: session.preferences,
        
        fields: [
            field theme: {
                type: "select",
                options: ["light", "dark", "auto"],
                value: session.preferences.theme
            },
            field language: {
                type: "select", 
                options: ["en", "es", "fr", "de"],
                value: session.preferences.language
            },
            field notifications: {
                type: "checkbox",
                checked: session.preferences.notifications
            }
        ]
        
        on_submit: {
            update_session: "preferences" = form,
            run_chain: "SaveUserPreferences",
            show_toast: "Preferences saved"
        }
    }
}

// Session-based cart management
page "ShoppingCart" at "/cart" {
    realtime: true  // Real-time cart sync across tabs
    
    show div "Cart Items" {
        children: [
            show list {
                from: session.cart.items,
                item as cart_item:
                    show div {
                        style: {
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "16px",
                            borderBottom: "1px solid #e5e7eb"
                        }
                        
                        children: [
                            show div {
                                children: [
                                    show text cart_item.name {
                                        style: {fontWeight: "600"}
                                    },
                                    show text "${cart_item.price}" {
                                        style: {color: "#6b7280"}
                                    }
                                ]
                            },
                            
                            show div {
                                style: {display: "flex", alignItems: "center", gap: "8px"}
                                children: [
                                    show button "-" {
                                        on_click: {
                                            update_session: "cart.items[{cart_item.id}].quantity" = cart_item.quantity - 1,
                                            if cart_item.quantity <= 1:
                                                remove_from_session: "cart.items[{cart_item.id}]",
                                            broadcast: "cart_updated",
                                            refresh: true
                                        }
                                    },
                                    show text cart_item.quantity,
                                    show button "+" {
                                        on_click: {
                                            update_session: "cart.items[{cart_item.id}].quantity" = cart_item.quantity + 1,
                                            broadcast: "cart_updated",
                                            refresh: true
                                        }
                                    }
                                ]
                            }
                        ]
                    }
            }
        ]
    }
    
    show div "Cart Total" {
        style: {padding: "16px", borderTop: "2px solid #e5e7eb", textAlign: "right"}
        children: [
            show text "Total: ${session.cart.total}" {
                style: {fontSize: "20px", fontWeight: "700"}
            }
        ]
    }
}
```

## Error Handling & Logging

### Comprehensive Error Management

```namel3ss
// Global error handling configuration
error_config {
    log_level: "info",  // debug, info, warn, error
    log_destination: "file",  // console, file, remote
    log_format: "json",
    
    // Error categorization
    error_types: {
        validation: {
            user_facing: true,
            log_level: "warn",
            retry_policy: "none"
        },
        network: {
            user_facing: false,
            log_level: "error", 
            retry_policy: "exponential_backoff",
            max_retries: 3
        },
        database: {
            user_facing: false,
            log_level: "error",
            retry_policy: "immediate",
            max_retries: 1
        }
    }
}

// Chain with error handling
chain "ProcessPayment" {
    input: payment_data
    error_handling: true
    
    step validate_payment:
        try:
            if !payment_data.card_token:
                throw ValidationError("Payment method required")
            if payment_data.amount <= 0:
                throw ValidationError("Amount must be greater than zero")
        catch ValidationError as e:
            log_warn("Payment validation failed", {error: e.message, user_id: session.user_id})
            return {success: false, error: e.message, error_type: "validation"}
    
    step charge_card:
        try:
            payment_result = payment_gateway.charge(payment_data)
            if !payment_result.success:
                throw PaymentError(payment_result.error_message)
        catch PaymentError as e:
            log_error("Payment processing failed", {
                error: e.message,
                payment_data: payment_data,
                user_id: session.user_id
            })
            return {success: false, error: "Payment processing failed", error_type: "payment"}
        catch NetworkError as e:
            log_error("Payment gateway unreachable", {error: e.message})
            return {success: false, error: "Service temporarily unavailable", error_type: "network"}
    
    step record_transaction:
        try:
            transaction = database.insert("transactions", {
                user_id: session.user_id,
                amount: payment_data.amount,
                status: "completed",
                gateway_transaction_id: payment_result.transaction_id
            })
        catch DatabaseError as e:
            // Payment succeeded but recording failed - needs manual reconciliation
            log_error("Transaction recording failed after successful payment", {
                error: e.message,
                payment_result: payment_result,
                user_id: session.user_id
            })
            alert_ops_team("Payment reconciliation needed", {
                payment_result: payment_result,
                user_id: session.user_id
            })
            // Still return success to user since payment went through
    
    output: {success: true, transaction_id: payment_result.transaction_id}
}

// Page with error handling UI
page "PaymentPage" at "/payment" {
    memory: payment_errors = [], payment_loading = false
    
    show form "Payment Form" {
        fields: [
            field amount: {
                type: "number",
                required: true,
                min: 0.01,
                validation: {
                    custom: amount > 10000 ? "Amount cannot exceed $10,000" : null
                }
            },
            field card_token: {type: "hidden"}  // From card widget
        ]
        
        on_submit: {
            clear_memory: "payment_errors",
            update_memory: "payment_loading" = true,
            
            try:
                run_chain: "ProcessPayment",
                show_toast: "Payment successful!",
                navigate: "/success"
            catch ValidationError as e:
                update_memory: "payment_errors" = [e.message],
                show_toast: e.message, "error"
            catch PaymentError as e:
                update_memory: "payment_errors" = [e.message],
                show_toast: e.message, "error"
            catch NetworkError as e:
                update_memory: "payment_errors" = ["Service temporarily unavailable. Please try again."],
                show_toast: "Service temporarily unavailable", "error"
            finally:
                update_memory: "payment_loading" = false
        }
    }
    
    // Error display
    if payment_errors.length > 0:
        show div "Error Messages" {
            style: {
                backgroundColor: "#fef2f2",
                border: "1px solid #fecaca", 
                borderRadius: "8px",
                padding: "12px",
                marginBottom: "16px"
            }
            
            children: [
                show list {
                    from: payment_errors,
                    item as error:
                        show text error {
                            style: {color: "#dc2626", fontSize: "14px"}
                        }
                }
            ]
        }
    
    // Loading state
    if payment_loading:
        show div "Processing..." {
            style: {
                textAlign: "center",
                padding: "32px",
                backgroundColor: "#f8fafc"
            }
        }
}
```

### System Monitoring

```namel3ss
// Monitoring and observability
monitor "ApplicationHealth" {
    metrics: {
        request_count: "counter",
        response_time: "histogram",
        error_rate: "gauge",
        active_users: "gauge"
    }
    
    alerts: [
        {
            name: "high_error_rate",
            condition: "error_rate > 0.05",
            notification: "slack",
            message: "Error rate above 5%"
        },
        {
            name: "slow_response",
            condition: "response_time.p95 > 1000",
            notification: "email",
            message: "95th percentile response time above 1 second"
        }
    ]
}

// Health check endpoint
page "HealthCheck" at "/health" {
    api_only: true
    
    health_checks: [
        {
            name: "database",
            check: database.ping(),
            timeout: "5s"
        },
        {
            name: "redis",
            check: redis.ping(),
            timeout: "2s"
        },
        {
            name: "payment_gateway", 
            check: payment_gateway.health(),
            timeout: "10s"
        }
    ]
    
    response: {
        status: all(health_checks.map(check => check.healthy)) ? "healthy" : "unhealthy",
        checks: health_checks,
        timestamp: now(),
        version: app.version
    }
}
```

## Production Deployment

### Environment Configuration

```namel3ss
// Multi-environment configuration
environments {
    development: {
        database: {
            host: "localhost",
            port: 5432,
            database: "namel3ss_dev",
            ssl: false
        },
        redis: {
            host: "localhost",
            port: 6379
        },
        debug: true,
        cors_origins: ["http://localhost:3000"]
    },
    
    staging: {
        database: {
            host: env("DATABASE_HOST"),
            port: env("DATABASE_PORT", 5432),
            database: env("DATABASE_NAME"),
            ssl: true
        },
        redis: {
            host: env("REDIS_HOST"),
            port: env("REDIS_PORT", 6379),
            password: env("REDIS_PASSWORD")
        },
        debug: false,
        cors_origins: ["https://staging.example.com"]
    },
    
    production: {
        database: {
            host: env("DATABASE_HOST"),
            port: env("DATABASE_PORT", 5432),
            database: env("DATABASE_NAME"),
            ssl: true,
            pool_size: 20
        },
        redis: {
            cluster: env("REDIS_CLUSTER_ENDPOINT"),
            password: env("REDIS_PASSWORD")
        },
        debug: false,
        cors_origins: ["https://app.example.com"],
        rate_limiting: {
            requests_per_minute: 1000,
            burst_size: 100
        }
    }
}

// Security configuration
security {
    authentication: {
        provider: "oauth2",
        jwt_secret: env("JWT_SECRET"),
        token_expiry: "1 hour",
        refresh_token_expiry: "30 days"
    },
    
    authorization: {
        default_role: "user",
        roles: {
            admin: ["read:all", "write:all", "delete:all"],
            moderator: ["read:all", "write:posts", "moderate:comments"],
            user: ["read:public", "write:own"]
        }
    },
    
    rate_limiting: {
        global: "1000/hour",
        per_user: "100/hour",
        login_attempts: "5/hour"
    },
    
    content_security_policy: {
        default_src: "'self'",
        script_src: "'self' 'unsafe-inline' https://cdn.example.com",
        img_src: "'self' data: https:",
        style_src: "'self' 'unsafe-inline'"
    }
}
```

### Scaling and Performance

```namel3ss
// Performance optimization
performance {
    caching: {
        strategy: "redis",
        ttl_default: "1 hour",
        
        policies: {
            "user_profile": "24 hours",
            "product_catalog": "4 hours", 
            "real_time_data": "30 seconds"
        }
    },
    
    database: {
        connection_pooling: true,
        pool_size: 20,
        query_timeout: "30 seconds",
        
        read_replicas: {
            enabled: true,
            urls: [
                env("DATABASE_READ_REPLICA_1"),
                env("DATABASE_READ_REPLICA_2")
            ]
        }
    },
    
    background_jobs: {
        provider: "celery",
        broker: "redis",
        
        queues: {
            default: {workers: 4},
            email: {workers: 2}, 
            analytics: {workers: 1}
        }
    }
}

// Background job processing
background_job "SendEmail" {
    queue: "email",
    retry_policy: {
        max_retries: 3,
        backoff: "exponential",
        max_delay: "1 hour"
    }
    
    task: {
        email_service.send({
            to: job.data.recipient,
            template: job.data.template,
            data: job.data.template_data
        })
    }
}

// Use background jobs in chains
chain "UserRegistration" {
    input: user_data
    
    step create_user:
        user = database.insert("users", user_data)
    
    step send_welcome_email:
        // Queue email for background processing
        queue_job("SendEmail", {
            recipient: user.email,
            template: "welcome",
            template_data: {name: user.name, activation_link: generate_activation_link(user.id)}
        })
    
    step track_registration:
        analytics.track("user_registered", {
            user_id: user.id,
            source: user_data.source || "direct"
        })
    
    output: {user_id: user.id}
}
```

This comprehensive guide demonstrates how Namel3ss provides enterprise-grade backend features, robust error handling, session management, and production deployment capabilities that make it suitable for building scalable, reliable applications.