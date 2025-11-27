if exists("b:current_syntax")
  finish
endif

" Highlight single-line Namel3ss comments with italic gray text and vivid emoji
syntax match namel3ssComment /^\s*#\s\zs\S.*$/ contains=namel3ssCommentEmoji
syntax match namel3ssCommentEmoji /\v#\s\zs\S/ contained

hi def Namel3ssComment gui=italic cterm=italic guifg=#9CA3AF
hi def link namel3ssComment Comment
hi def Namel3ssCommentEmoji guifg=#f59e0b

let b:current_syntax = "namel3ss"
