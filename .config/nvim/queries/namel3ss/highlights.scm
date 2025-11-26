; Neovim Treesitter Highlights for Namel3ss
; For use with Neovim 0.8+ and nvim-treesitter

; Comments
[
  (line_comment)
  (block_comment)
] @comment

; Keywords
[
  "app"
  "page"
  "action"
  "if"
  "else"
  "for"
  "while"
  "in"
  "llm"
  "prompt"
  "memory"
  "frame"
  "dataset"
  "function"
  "return"
  "import"
  "from"
  "as"
  "with"
  "model"
  "tool"
  "context"
  "eval"
  "train"
  "test"
  "validate"
  "deploy"
  "monitor"
  "log"
  "trace"
] @keyword

; Booleans
[
  "true"
  "false"
  "True"
  "False"
  "null"
  "None"
] @boolean

; UI Components
[
  "show_text"
  "show_table"
  "show_form"
  "show_image"
  "show_button"
  "show_input"
  "show_select"
  "show_checkbox"
  "show_radio"
  "stack"
  "grid"
  "modal"
  "toast"
  "tabs"
  "accordion"
  "card"
] @function.builtin

; AI Semantic Components
[
  "chat_thread"
  "agent_panel"
  "tool_call_view"
  "log_view"
  "evaluation_result"
  "diff_view"
] @type.builtin

; Properties
[
  "title"
  "description"
  "label"
  "value"
  "placeholder"
  "width"
  "height"
  "padding"
  "margin"
  "background"
  "color"
  "on_click"
  "on_change"
  "on_submit"
  "validation"
  "required"
  "data"
  "columns"
  "rows"
  "items"
  "source"
  "destination"
  "style"
  "class"
  "id"
  "name"
  "type"
  "href"
  "src"
  "alt"
  "open"
  "is_open"
  "on_close"
  "variant"
  "position"
  "duration"
] @property

; AI-specific Properties
[
  "messages_binding"
  "agent_binding"
  "tool_calls_binding"
  "log_entries_binding"
  "metric_binding"
  "left_binding"
  "right_binding"
  "show_tokens"
  "streaming_enabled"
  "auto_scroll"
  "show_metadata"
  "show_input"
  "show_status"
  "show_memory"
  "show_context"
  "show_tool_output"
  "show_timestamp"
  "diff_mode"
  "syntax_highlight"
  "editable"
  "line_numbers"
  "word_wrap"
] @attribute

; Strings
(string) @string
(string_interpolation) @string.special

; Numbers
(integer) @number
(float) @number

; Operators
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  "&&"
  "||"
  "and"
  "or"
  "not"
  "="
] @operator

; Identifiers and Functions
(identifier) @variable
(function_call) @function
(binding) @variable.member

; Punctuation
[
  "("
  ")"
  "{"
  "}"
  "["
  "]"
  ":"
  ","
  "."
] @punctuation.delimiter
