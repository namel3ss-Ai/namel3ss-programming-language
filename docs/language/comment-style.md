# Namel3ss comment style

Single marker comments keep the language consistent across every editor, parser, and toolchain.

## Syntax
- Format: `# <emoji?> <text>`
- Required regex: `^#\s\S.*` (emoji-friendly: `^#\s[\p{Emoji}]?\s?.*`)
- Examples:
  - `# 💬 This section monitors model metrics`
  - `# ⚠️ This action impacts production data`
  - `# 🔒 Admins only`

## Style
- Font: italic
- Color: gray `#9CA3AF`
- Emoji: full color (not dimmed)
- Background: transparent
- Token name used by editors: `comment.line.namel3ss` + `constant.character.emoji.namel3ss`

## Parser and IR behavior
- Comments are ignored for IR generation and never create AST nodes.
- All valid comments are captured as metadata (`Module.comments`) with text, emoji (when present), line, and column.
- Invalid markers raise syntax errors:
  - `#⚠️Missing space`
  - `## double markers`
  - `//` or `/* */`

## Editor support
- VS Code / Cursor: `editor/vscode/syntaxes/namel3ss.tmLanguage.json` and `themes/namel3ss-comments-color-theme.json`
- JetBrains: `editor-support/jetbrains/namel3ss-comment.xml` and `Namel3ssCommentAnnotator.kt`
- Neovim/Vim: `editor-support/nvim/queries/namel3ss/highlights.scm` and `editor-support/vim/syntax/namel3ss.vim`
- Sublime Text: `Namel3ss.sublime-syntax` + `editor-support/sublime/namel3ss-comments.sublime-color-scheme`
- GitHub: `.gitattributes` + `linguist-languages.yml`

## Regex tokens for highlighters
- Primary: `^#\s\S.*`
- Emoji intent: `^#\s[\p{Emoji}]?\s?.*`
- Token identifier: `Namel3ssComment`

## Generators and tooling
- Code generators (React/TS, Python backend) emit no comments unless explicitly configured.
- Linter includes an optional strict rule for enforcing the `# <emoji?> <text>` style (`CommentStyleRule` or `get_default_rules(strict_comment_style=True)`).
- Language server surfaces comment metadata for documentation, hover text, and structure outlines.

## Test cases
- Valid: `# 💬 Comment`
- Invalid: `#⚠️Missing space`
- Invalid: `## Wrong marker`
- Invalid: `// Not allowed`
- Invalid: `/* Not allowed */`
