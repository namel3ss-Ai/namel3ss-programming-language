# Namel3ss editor support

Centralized snippets to keep the single-line `#` comment style consistent across editors.

- VS Code / Cursor: `editor/vscode/…` grammar + `themes/namel3ss-comments-color-theme.json`
- JetBrains: `editor-support/jetbrains/namel3ss-comment.xml` + `Namel3ssCommentAnnotator.kt`
- Sublime Text: `Namel3ss.sublime-syntax` + `editor-support/sublime/namel3ss-comments.sublime-color-scheme`
- Vim/Neovim: `editor-support/vim/syntax/namel3ss.vim` and Treesitter query `editor-support/nvim/queries/namel3ss/highlights.scm`
- GitHub Linguist: `linguist-languages.yml` + `.gitattributes`

The shared regex for comments is `^#\s\S.*` (emoji-friendly via `^#\s[\p{Emoji}]?\s?.*`). Comments render italic gray (`#9CA3AF`) with bright emojis left unstyled.
