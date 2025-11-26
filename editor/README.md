# Universal File Type Associations for Namel3ss

This directory contains configuration files for various editors to recognize `.n3` and `.ai` files as Namel3ss language files with full syntax highlighting.

## ✨ Syntax Highlighting

Namel3ss now features comprehensive syntax highlighting with color-coded keywords, components, properties, strings, numbers, and more—just like Python and other modern languages.

**See [SYNTAX_HIGHLIGHTING.md](../SYNTAX_HIGHLIGHTING.md) for detailed installation and customization.**

## Supported Editors

### ✅ All Editors (.editorconfig)
- **File**: `../.editorconfig` (root of project)
- **Automatic**: Works with VS Code, IntelliJ, Sublime Text, Atom, Vim, Emacs, and many more
- **No setup required**: Just open the project

### ✅ VS Code
- **Location**: `../demo-vscode-extension/`
- **Setup**: Install the extension or reload window (Ctrl+R / Cmd+R)
- **Features**: 
  - Full syntax highlighting with 60+ keywords
  - AI components color-coded separately
  - Properties and bindings highlighted
  - String interpolation support
  - Comment syntax (# // /* */)
  - File icons for `.ai` files
- **Grammar**: `demo-vscode-extension/syntaxes/namel3ss.tmLanguage.json`

### ✅ Vim/Vi
- **Files**: 
  - `.vim/syntax/namel3ss.vim` (syntax rules)
  - `.vim/ftdetect/namel3ss.vim` (file detection)
- **Setup**: Copy to `~/.vim/syntax/` and `~/.vim/ftdetect/`
- **Features**: 
  - Keywords, components, AI components
  - Properties with dedicated highlight groups
  - String, number, boolean highlighting
  - Comment syntax
  - Operator highlighting
  - Binding dotted notation

### ✅ Neovim
- **File**: `.config/nvim/queries/namel3ss/highlights.scm`
- **Setup**: 
  1. Copy to `~/.config/nvim/queries/namel3ss/`
  2. Add filetype detection to `init.lua`
- **Features**: 
  - Modern Treesitter-style highlighting
  - All language constructs color-coded
  - Semantic token support
  - Icon support with nvim-web-devicons

### ✅ Sublime Text
- **File**: `Namel3ss.sublime-syntax` (YAML format)
- **Setup**: Copy to `Packages/User/`
- **Features**: 
  - Complete syntax definition with contexts
  - String interpolation `{variable.path}`
  - Triple-quoted strings
  - All component types
  - Property highlighting

### ✅ JetBrains IDEs (IntelliJ, PyCharm, WebStorm)
- **Automatic**: Uses `.editorconfig` for basic settings
- **Full Support**: Would require a custom plugin (can be created if needed)

### ✅ Emacs
- **Automatic**: Uses `.editorconfig` via `editorconfig-emacs` package
- **Additional**: Can create custom mode if needed

### ✅ Atom
- **Automatic**: Uses `.editorconfig` via `editorconfig` package
- **Additional**: File Icons package will use GitHub Linguist settings

## Git Integration

### GitHub/GitLab Language Detection
- **File**: `../.gitattributes`
- **Setup**: Automatic when file is in repository
- **Features**: 
  - Marks `.n3` and `.ai` files as Namel3ss language
  - Enables GitHub syntax highlighting
  - Consistent line endings (LF)
  - Proper language statistics

## File Extension Support

Both `.n3` and `.ai` extensions are supported across all editors:

- **`.n3`**: Original Namel3ss file extension
- **`.ai`**: AI-focused Namel3ss files

## Installation Quick Start

### Automatic (Recommended)
1. Clone/open the Namel3ss repository
2. Files are automatically detected via `.editorconfig` and `.gitattributes`

### Editor-Specific Setup

**Vim/Neovim**:
```bash
# Vim
mkdir -p ~/.vim/ftdetect
cp editor/vim/ftdetect/namel3ss.vim ~/.vim/ftdetect/

# Neovim
mkdir -p ~/.config/nvim/lua
cp editor/nvim/namel3ss.lua ~/.config/nvim/lua/
echo "require('namel3ss')" >> ~/.config/nvim/init.lua
```

**Sublime Text**:
```bash
# Find Sublime Text packages directory
# Windows: %APPDATA%/Sublime Text/Packages/User/
# macOS: ~/Library/Application Support/Sublime Text/Packages/User/
# Linux: ~/.config/sublime-text/Packages/User/

cp editor/sublime/Namel3ss.sublime-syntax [PACKAGES_DIR]/User/
```

**VS Code**:
```bash
# Install from marketplace or load locally
code --install-extension namel3ss-ai-assistant
```

## Features by Editor

| Editor | File Detection | Syntax Highlighting | Indentation | Icon | Auto-complete |
|--------|---------------|---------------------|-------------|------|---------------|
| VS Code | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vim | ✅ | ⚠️* | ✅ | ❌ | ⚠️* |
| Neovim | ✅ | ⚠️* | ✅ | ✅** | ⚠️* |
| Sublime Text | ✅ | ⚠️* | ✅ | ⚠️*** | ❌ |
| IntelliJ/PyCharm | ✅ | ❌ | ✅ | ❌ | ❌ |
| Atom | ✅ | ⚠️* | ✅ | ⚠️*** | ❌ |
| Emacs | ✅ | ❌ | ✅ | ❌ | ❌ |

\* Requires syntax file (can be created if needed)  
\** Requires nvim-web-devicons plugin  
\*** Via file-icons or similar packages

## MIME Type

Namel3ss files use the custom MIME type:
- `text/x-namel3ss` for `.n3` files
- `text/x-namel3ss` for `.ai` files

This is recognized by:
- Web servers for proper content-type headers
- Git hosting platforms (GitHub, GitLab, Bitbucket)
- File managers with MIME type support

## Contributing

To add support for additional editors:
1. Create configuration file in appropriate `editor/[editor-name]/` directory
2. Update this README with installation instructions
3. Test with actual editor installation
4. Submit pull request

## Need Help?

If your editor isn't listed or you need help setting up:
1. Check if your editor supports `.editorconfig` (most modern editors do)
2. Open an issue describing your editor and needs
3. Check editor's plugin/extension marketplace for Namel3ss support
