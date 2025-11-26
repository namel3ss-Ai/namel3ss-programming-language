# Vim configuration for Namel3ss (.n3 and .ai files)
# Place this file at: ~/.vim/ftdetect/namel3ss.vim
# Or in your Neovim config: ~/.config/nvim/ftdetect/namel3ss.vim

# Detect Namel3ss files by extension
au BufRead,BufNewFile *.n3 set filetype=namel3ss
au BufRead,BufNewFile *.ai set filetype=namel3ss

# Set syntax highlighting (if namel3ss.vim syntax file exists)
au FileType namel3ss set syntax=namel3ss

# Set indentation for Namel3ss files
au FileType namel3ss setlocal expandtab shiftwidth=2 softtabstop=2 tabstop=2

# Enable line numbers
au FileType namel3ss setlocal number

# Set text width
au FileType namel3ss setlocal textwidth=120

# Enable syntax folding
au FileType namel3ss setlocal foldmethod=syntax
