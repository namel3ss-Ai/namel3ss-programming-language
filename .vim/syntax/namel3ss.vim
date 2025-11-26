" Vim syntax file for Namel3ss (.ai files)
" Language: Namel3ss
" Maintainer: Namel3ss Project
" Latest Revision: 2024

if exists("b:current_syntax")
  finish
endif

" Comments
syn match namel3ssComment "#.*$"
syn match namel3ssComment "//.*$"
syn region namel3ssBlockComment start="/\*" end="\*/"

" Keywords
syn keyword namel3ssKeyword app page action if else for while in llm prompt memory frame dataset
syn keyword namel3ssKeyword function return import from as with model tool context
syn keyword namel3ssKeyword eval train test validate deploy monitor log trace
syn keyword namel3ssBoolean true false True False null None

" UI Components
syn keyword namel3ssComponent show_text show_table show_form show_image stack grid
syn keyword namel3ssComponent show_button show_input show_select show_checkbox show_radio
syn keyword namel3ssComponent modal toast tabs accordion card

" AI Semantic Components
syn keyword namel3ssAIComponent chat_thread agent_panel tool_call_view log_view
syn keyword namel3ssAIComponent evaluation_result diff_view

" Properties (common attributes)
syn keyword namel3ssProperty title description label value placeholder
syn keyword namel3ssProperty width height padding margin background color
syn keyword namel3ssProperty on_click on_change on_submit validation required
syn keyword namel3ssProperty data columns rows items source destination
syn keyword namel3ssProperty style class id name type href src alt
syn keyword namel3ssProperty open is_open on_close variant position duration

" AI-specific properties
syn keyword namel3ssAIProperty messages_binding agent_binding tool_calls_binding
syn keyword namel3ssAIProperty log_entries_binding metric_binding left_binding right_binding
syn keyword namel3ssAIProperty show_tokens streaming_enabled auto_scroll show_metadata
syn keyword namel3ssAIProperty show_input show_status show_memory show_context
syn keyword namel3ssAIProperty show_tool_output show_timestamp diff_mode syntax_highlight
syn keyword namel3ssAIProperty editable line_numbers word_wrap

" Strings
syn region namel3ssString start='"' end='"' contains=namel3ssInterpolation
syn region namel3ssString start="'" end="'" contains=namel3ssInterpolation
syn region namel3ssStringTriple start='"""' end='"""' contains=namel3ssInterpolation
syn region namel3ssStringTriple start="'''" end="'''" contains=namel3ssInterpolation
syn match namel3ssInterpolation "{[^}]*}" contained

" Numbers
syn match namel3ssNumber "\<\d\+\>"
syn match namel3ssNumber "\<\d\+\.\d\+\>"
syn match namel3ssNumber "\<\d\+[eE][+-]\?\d\+\>"
syn match namel3ssNumber "\<\d\+\.\d\+[eE][+-]\?\d\+\>"

" Operators
syn match namel3ssOperator "[+\-*/%]"
syn match namel3ssOperator "[<>=!]="
syn match namel3ssOperator "[<>]"
syn match namel3ssOperator "&&"
syn match namel3ssOperator "||"
syn keyword namel3ssOperator and or not

" Bindings (dotted notation)
syn match namel3ssBinding "\w\+\(\.\w\+\)\+"

" Functions
syn match namel3ssFunction "\<\w\+\>\s*("me=e-1

" Highlight Links
hi def link namel3ssComment Comment
hi def link namel3ssBlockComment Comment
hi def link namel3ssKeyword Keyword
hi def link namel3ssBoolean Boolean
hi def link namel3ssComponent Function
hi def link namel3ssAIComponent Special
hi def link namel3ssProperty Identifier
hi def link namel3ssAIProperty Type
hi def link namel3ssString String
hi def link namel3ssStringTriple String
hi def link namel3ssInterpolation Special
hi def link namel3ssNumber Number
hi def link namel3ssOperator Operator
hi def link namel3ssBinding Variable
hi def link namel3ssFunction Function

let b:current_syntax = "namel3ss"
