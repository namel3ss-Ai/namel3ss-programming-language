# Namel3ss Troubleshooting Guide

This guide addresses common issues encountered when developing with namel3ss, based on real user experiences.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Syntax Errors](#syntax-errors)
3. [JavaScript/ES6 Limitations](#javascript-limitations)
4. [Indentation Issues](#indentation-issues)
5. [Runtime Issues](#runtime-issues)
6. [Best Practices](#best-practices)

---

## Installation Issues

### Problem: `namel3ss` command not found
```bash
$ namel3ss generate chatbot.ai chatbot_output
zsh: command not found: namel3ss
```

**Solutions:**
1. **Clear cache and reinstall:**
   ```bash
   pip uninstall namel3ss -y
   pip cache purge
   pip install namel3ss==0.5.1
   ```

2. **Verify installation:**
   ```bash
   namel3ss --version  # Should show 0.5.1
   ```

3. **Development installation:**
   ```bash
   git clone https://github.com/namel3ss-Ai/namel3ss-programming-language.git
   cd namel3ss-programming-language
   pip install -e .
   ```

---

## Syntax Errors

### 1. Prompt Definition Syntax

❌ **Wrong:**
```n3
prompt ChatbotPrompt {
  // Syntax error - no quotes
}
```

✅ **Correct:**
```n3
prompt "chatbot_prompt":
  input:
    message: text
    history: conversation
  output:
    response: text
  using model "gpt-3.5-turbo":
    """
    You are a helpful assistant.
    User: {{message}}
    """
```

### 2. Chain Step Prompt References

❌ **Wrong:**
```n3
step generate_response:
  prompt: "chatbot_prompt"  # Cannot reference prompts by name
```

✅ **Correct:**
```n3
step generate_response:
  model: chat_model
  prompt: |
    You are a helpful assistant.
    User message: {{message}}
    History: {{history}}
```

### 3. Multi-line Template Strings

❌ **Wrong (causes indentation errors):**
```n3
prompt: |
      You are a helpful assistant.
    User message: {{message}}  # Inconsistent indentation
```

✅ **Correct:**
```n3
prompt: |
  You are a helpful assistant.
  User message: {{message}}
  History: {{history}}
```

---

## JavaScript Limitations

Namel3ss has limited support for modern JavaScript features in `layout:` blocks.

### Unsupported Features

❌ **Template Literals (Backticks):**
```javascript
// This will cause syntax errors
const message = `Error: ${error.message}`;
```

❌ **Optional Chaining:**
```javascript
// This will cause syntax errors  
const detail = error.response?.data?.detail;
```

❌ **Arrow Functions (in some contexts):**
```javascript
// May cause issues
array.map(item => item.name);
```

### Workarounds

✅ **Use String Concatenation:**
```javascript
const message = 'Error: ' + error.message;
```

✅ **Use Traditional Null Checking:**
```javascript
const detail = error.response && error.response.data && error.response.data.detail;
```

✅ **Use Function Expressions:**
```javascript
array.map(function(item) { return item.name; });
```

---

## Indentation Issues

### Common Indentation Problems

1. **Inconsistent spacing in layout blocks**
2. **Mixed tabs and spaces**
3. **CSS/JavaScript indentation within HTML**

### Best Practices

✅ **Consistent Indentation in Layout:**
```n3
layout: |
  <!DOCTYPE html>
  <html>
    <head>
      <style>
        body {
          margin: 0;
          padding: 20px;
        }
        
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>My App</h1>
      </div>
      
      <script>
        const app = {
          data() {
            return {
              message: 'Hello World'
            };
          }
        };
      </script>
    </body>
  </html>
```

### Debugging Indentation

1. **Use consistent spacing (2 or 4 spaces)**
2. **Avoid mixing tabs and spaces**
3. **Check error line numbers carefully**
4. **Use a code editor with indentation guides**

---

## Runtime Issues

### Server Not Starting

**Common Causes & Solutions:**

1. **Missing Environment Variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export PORT=8000
   ```

2. **Port Already in Use:**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different port
   uvicorn main:app --port 8001
   ```

3. **Missing Dependencies:**
   ```bash
   pip install fastapi uvicorn openai httpx
   ```

4. **Syntax Errors Not Caught:**
   ```bash
   # Test syntax validation
   namel3ss build chatbot.ai --validate-only
   ```

### API Connection Issues

**Problem:** Frontend can't connect to backend API

**Solutions:**
1. **Check CORS configuration in generated FastAPI code**
2. **Verify API endpoints are generated correctly**
3. **Test API manually:**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello"}'
   ```

---

## Best Practices

### 1. File Structure
```
my-project/
├── chatbot.ai              # Main namel3ss file
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── generated/             # Generated code (don't edit)
    ├── backend/
    └── frontend/
```

### 2. Environment Setup
```bash
# .env file
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
PORT=8000
DEBUG=true
```

### 3. Model Configuration
```n3
# Always specify model parameters
model chat_model:
  provider: openai
  name: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30
```

### 4. Error Handling in Chains
```n3
chain "safe_chat":
  inputs:
    message: text
    
  steps:
    - step validate_input:
        condition: "len(message.strip()) > 0"
        error_message: "Message cannot be empty"
        
    - step generate_response:
        model: chat_model
        prompt: |
          You are a helpful assistant.
          User: {{message}}
        fallback: "I'm sorry, I couldn't process your request right now."
```

### 5. Memory Management
```n3
# Use appropriate memory scopes
memory "user_preferences":
  scope: user
  kind: key_value
  ttl: 86400  # 24 hours

memory "conversation_history":
  scope: conversation  
  kind: conversation
  max_items: 50
```

---

## Development Workflow

### 1. Start Small
```n3
# Begin with minimal example
model simple_model:
  provider: openai
  name: "gpt-3.5-turbo"

endpoint "/test" {
  method: get
  returns:
    message: "Hello World"
}

layout: |
  <!DOCTYPE html>
  <html><body><h1>It works!</h1></body></html>
```

### 2. Test Incrementally
```bash
# 1. Validate syntax
namel3ss build app.ai --validate-only

# 2. Generate code
namel3ss build app.ai --output generated

# 3. Run server
cd generated/backend
uvicorn main:app --reload --port 8000
```

### 3. Debug Step by Step
1. **Test model connectivity** (simple prompt)
2. **Add basic endpoints** (no AI)
3. **Add simple AI chain** (one step)
4. **Add memory/state**
5. **Add complex frontend**

---

## Getting Help

### Error Reporting Checklist
- [ ] Include full error message and stack trace
- [ ] Provide minimal reproducible example
- [ ] List namel3ss version (`namel3ss --version`)
- [ ] Include environment details (Python version, OS)
- [ ] Show relevant parts of your `.ai` file

### Resources
- [Documentation](https://github.com/namel3ss-Ai/namel3ss-programming-language/blob/main/NAMEL3SS_DOCUMENTATION.md)
- [API Reference](https://github.com/namel3ss-Ai/namel3ss-programming-language/blob/main/API_REFERENCE.md)
- [Examples](https://github.com/namel3ss-Ai/namel3ss-programming-language/tree/main/examples)
- [Issues](https://github.com/namel3ss-Ai/namel3ss-programming-language/issues)