# Namel3ss Syntax Highlighting Quick Reference

## Color Scheme Preview

### Keywords (Blue/Purple)
```namel3ss
app page action if else for while in
llm prompt memory frame dataset
function return import from as with
model tool context eval train test
```

### UI Components (Yellow/Bold)
```namel3ss
show_text show_table show_form show_image
show_button show_input show_select
stack grid modal toast tabs accordion card
```

### AI Semantic Components (Orange/Special)
```namel3ss
chat_thread agent_panel tool_call_view
log_view evaluation_result diff_view
```

### Properties (Cyan/Light Blue)
```namel3ss
title description label value placeholder
width height padding margin background color
on_click on_change on_submit validation
data columns rows items source destination
```

### AI Properties (Cyan/Attribute)
```namel3ss
messages_binding agent_binding tool_calls_binding
show_tokens streaming_enabled auto_scroll
show_metadata show_input show_status
diff_mode syntax_highlight editable
```

### Strings (Green/Orange)
```namel3ss
"double quoted string"
'single quoted string'
"""triple quoted
multi-line string"""
"interpolation: {variable.path}"
```

### Numbers (Teal/Cyan)
```namel3ss
42            # integer
3.14          # float
2.5e10        # scientific notation
-99.99
```

### Booleans (Purple/Magenta)
```namel3ss
true false True False
null None
```

### Comments (Gray/Muted)
```namel3ss
# Single line comment (shell style)
// Single line comment (C++ style)
/* Block comment
   can span multiple lines */
```

### Operators (White/Default)
```namel3ss
# Arithmetic
+ - * / %

# Comparison
== != < > <= >=

# Logical
&& || and or not

# Assignment
=
```

### Bindings (Default/Variable)
```namel3ss
agent_run.conversation
data.items.0.name
user.profile.settings.theme
ctx:steps.classify.result
```

### Functions (Yellow/Bold)
```namel3ss
get_user_data()
process_tickets(urgency)
format_response(template)
```

## Complete Example with Colors

```namel3ss
# Complete syntax demonstration
app "AI Dashboard"                     # keyword + string

page "Analytics" at "/analytics":      # keyword + string
  # UI Components (yellow)
  show_text "title":                   # component + property
    value: "Dashboard"                 # property + string
    style: "heading"                   # property + string
  
  # AI Component (orange)
  chat_thread "conversation":          # ai-component
    messages_binding: agent.messages   # ai-property + binding
    streaming_enabled: true            # ai-property + boolean
    show_tokens: false                 # ai-property + boolean
    auto_scroll: true                  # ai-property + boolean
  
  # Control flow (blue)
  if user.role == "admin":             # keyword + binding + operator + string
    show_button "export":              # component
      on_click: export_data()          # property + function
  
  # AI Component (orange)
  agent_panel "status":                # ai-component
    agent_binding: current_agent       # ai-property + binding
    show_metrics: true                 # ai-property + boolean
    show_cost: 0.05                    # ai-property + number
```

## Scope Mapping

| Element | TextMate Scope | Vim Highlight | Sublime Scope |
|---------|---------------|---------------|---------------|
| Keywords | `keyword.control` | `Keyword` | `keyword.control.namel3ss` |
| UI Components | `entity.name.tag` | `Function` | `entity.name.tag.component` |
| AI Components | `entity.name.tag.ai` | `Special` | `entity.name.tag.ai` |
| Properties | `variable.parameter` | `Identifier` | `variable.parameter.property` |
| AI Properties | `variable.parameter.ai` | `Type` | `variable.parameter.ai-property` |
| Strings | `string.quoted` | `String` | `string.quoted.double` |
| Numbers | `constant.numeric` | `Number` | `constant.numeric` |
| Booleans | `constant.language` | `Boolean` | `constant.language` |
| Comments | `comment.line` | `Comment` | `comment.line.number-sign` |
| Operators | `keyword.operator` | `Operator` | `keyword.operator.arithmetic` |
| Bindings | `variable.other.binding` | `Variable` | `variable.other.binding` |
| Functions | `entity.name.function` | `Function` | `entity.name.function` |

## Testing Your Setup

1. Open `examples/ai_components_demo.ai`
2. Verify these elements are colored:
   - ✅ `page` keyword in blue
   - ✅ `chat_thread` component in orange
   - ✅ `messages_binding` property in cyan
   - ✅ `"conversation_view"` string in green
   - ✅ `true` boolean in purple
   - ✅ `# Comment` in gray

3. If colors don't appear:
   - **VS Code**: Reload window (Ctrl+Shift+P → Reload Window)
   - **Vim**: `:syntax enable` then `:set filetype=namel3ss`
   - **Neovim**: Check filetype with `:set filetype?`
   - **Sublime**: View → Syntax → Namel3ss

## Customizing Colors

Colors depend on your editor's theme. The syntax defines semantic scopes that themes map to colors.

### VS Code Theme Mapping
Edit your theme's `.json` file to customize colors:

```json
{
  "tokenColors": [
    {
      "scope": "entity.name.tag.ai",
      "settings": {"foreground": "#FF6B35"}
    },
    {
      "scope": "variable.parameter.ai",
      "settings": {"foreground": "#4ECDC4"}
    }
  ]
}
```

### Vim Custom Highlights
Add to your `.vimrc`:

```vim
hi namel3ssAIComponent ctermfg=208 guifg=#FF6B35
hi namel3ssAIProperty ctermfg=80 guifg=#4ECDC4
```

### Neovim Custom Highlights
Add to your `init.lua`:

```lua
vim.api.nvim_set_hl(0, '@type.builtin', { fg = '#FF6B35' })
vim.api.nvim_set_hl(0, '@attribute', { fg = '#4ECDC4' })
```

## See Also

- [SYNTAX_HIGHLIGHTING.md](SYNTAX_HIGHLIGHTING.md) - Full installation guide
- [editor/README.md](editor/README.md) - Editor configuration
- [examples/ai_components_demo.ai](examples/ai_components_demo.ai) - Test file
