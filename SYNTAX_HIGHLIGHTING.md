# Namel3ss Syntax Highlighting

This directory contains syntax highlighting definitions for Namel3ss (.ai and .n3 files) across multiple editors.

## Features

The syntax highlighting includes:
- **Comments**: `#`, `//`, and `/* */` block comments
- **Keywords**: Control flow (if, else, for, while), declarations (app, page, action), AI primitives (llm, prompt, memory, dataset)
- **UI Components**: show_text, show_table, show_form, stack, grid, modal, toast, etc.
- **AI Components**: chat_thread, agent_panel, tool_call_view, log_view, evaluation_result, diff_view
- **Properties**: Standard (title, description, width, height) and AI-specific (messages_binding, streaming_enabled, show_tokens)
- **Strings**: Single, double, and triple-quoted with interpolation support `{variable.path}`
- **Numbers**: Integers, floats, and scientific notation
- **Booleans**: true, false, True, False, null, None
- **Operators**: Arithmetic, comparison, logical, and assignment
- **Bindings**: Dotted notation like `data.items.0.name`

## Installation

### VS Code

The syntax is automatically loaded from the VS Code extension:

```bash
cd demo-vscode-extension
code --install-extension .
```

Or if you're in VS Code, reload the window (Ctrl+R / Cmd+R) to apply changes.

### Vim

Copy the syntax files to your Vim runtime directory:

```bash
# Unix/Linux/macOS
mkdir -p ~/.vim/syntax ~/.vim/ftdetect
cp .vim/syntax/namel3ss.vim ~/.vim/syntax/
cp .vim/ftdetect/namel3ss.vim ~/.vim/ftdetect/

# Windows
mkdir %USERPROFILE%\vimfiles\syntax %USERPROFILE%\vimfiles\ftdetect
copy .vim\syntax\namel3ss.vim %USERPROFILE%\vimfiles\syntax\
copy .vim\ftdetect\namel3ss.vim %USERPROFILE%\vimfiles\ftdetect\
```

### Neovim (Treesitter)

Copy the highlights to your Neovim config:

```bash
# Unix/Linux/macOS
mkdir -p ~/.config/nvim/queries/namel3ss
cp .config/nvim/queries/namel3ss/highlights.scm ~/.config/nvim/queries/namel3ss/

# Windows
mkdir %LOCALAPPDATA%\nvim\queries\namel3ss
copy .config\nvim\queries\namel3ss\highlights.scm %LOCALAPPDATA%\nvim\queries\namel3ss\
```

Add to your `init.lua` or `init.vim`:

```lua
-- init.lua
vim.filetype.add({
  extension = {
    ai = 'namel3ss',
    n3 = 'namel3ss',
  }
})
```

### Sublime Text

Install the syntax file:

```bash
# Unix/Linux/macOS
cp Namel3ss.sublime-syntax ~/Library/Application\ Support/Sublime\ Text/Packages/User/

# Windows
copy Namel3ss.sublime-syntax %APPDATA%\Sublime Text\Packages\User\
```

Restart Sublime Text and open any `.ai` or `.n3` file.

## Color Scheme

The syntax uses standard TextMate/Sublime scopes that work with most color schemes:

| Element | Scope | Typical Color |
|---------|-------|---------------|
| Keywords | `keyword.control` | Blue/Purple |
| UI Components | `entity.name.tag` | Yellow/Bold |
| AI Components | `entity.name.tag.ai` | Orange/Special |
| Properties | `variable.parameter` | Cyan/Light Blue |
| Strings | `string.quoted` | Green/Orange |
| Numbers | `constant.numeric` | Teal/Cyan |
| Booleans | `constant.language` | Purple/Magenta |
| Comments | `comment.line` | Gray/Muted |
| Operators | `keyword.operator` | White/Default |
| Functions | `entity.name.function` | Yellow/Bold |

## Testing

Test the syntax highlighting with the example file:

```bash
# VS Code
code examples/ai_components_demo.ai

# Vim
vim examples/ai_components_demo.ai

# Neovim
nvim examples/ai_components_demo.ai

# Sublime Text
subl examples/ai_components_demo.ai
```

You should see:
- Keywords like `app`, `page`, `if` in blue/purple
- Components like `show_text`, `chat_thread` in yellow/orange
- Properties like `title`, `messages_binding` in cyan
- Strings in green, numbers in teal, comments in gray

## Troubleshooting

### VS Code: No colors appearing
1. Reload the window: `Ctrl+Shift+P` → "Developer: Reload Window"
2. Check file extension is `.ai` or `.n3`
3. Check bottom-right language selector shows "n3"

### Vim: No colors appearing
1. Ensure filetype plugin is enabled: `:filetype plugin on`
2. Check syntax is enabled: `:syntax enable`
3. Manually set filetype: `:set filetype=namel3ss`
4. Verify file is loaded: `:scriptnames` (should show namel3ss.vim)

### Neovim: No colors appearing
1. Check Treesitter is installed: `:checkhealth nvim-treesitter`
2. Verify filetype is set: `:set filetype?` (should show namel3ss)
3. Manually set filetype: `:set filetype=namel3ss`

### Sublime Text: No colors appearing
1. Check syntax is selected: View → Syntax → Namel3ss
2. Restart Sublime Text
3. Verify file extension is `.ai` or `.n3`

## Contributing

To add new keywords or components to the syntax:

1. **VS Code**: Edit `demo-vscode-extension/syntaxes/namel3ss.tmLanguage.json`
2. **Vim**: Edit `.vim/syntax/namel3ss.vim`
3. **Neovim**: Edit `.config/nvim/queries/namel3ss/highlights.scm`
4. **Sublime**: Edit `Namel3ss.sublime-syntax`

Keep all syntax files in sync to ensure consistent highlighting across editors.

## Examples

### AI Components Example

```namel3ss
# AI Semantic Components Demo
app AIDemo

page ChatDemo
  # Chat interface with real-time streaming
  chat_thread
    messages_binding: conversation.messages
    streaming_enabled: true
    show_tokens: true
    show_metadata: true
```

Should display with:
- `# AI Semantic Components Demo` in gray (comment)
- `app`, `page` in blue (keywords)
- `chat_thread` in orange (AI component)
- `messages_binding`, `streaming_enabled` in cyan (properties)
- `conversation.messages` in default (binding)
- `true` in purple (boolean)
- String values in green

## License

Same as the main Namel3ss project.
