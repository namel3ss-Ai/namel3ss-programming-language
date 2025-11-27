-- Neovim configuration for Namel3ss (.n3 and .ai files)
-- Place this in your Neovim config: ~/.config/nvim/lua/namel3ss.lua
-- Then add: require('namel3ss') to your init.lua

-- Detect Namel3ss files
vim.api.nvim_create_autocmd({"BufRead", "BufNewFile"}, {
  pattern = {"*.n3", "*.ai"},
  callback = function()
    vim.bo.filetype = "namel3ss"
  end,
})

-- Set Namel3ss file settings
vim.api.nvim_create_autocmd("FileType", {
  pattern = "namel3ss",
  callback = function()
    vim.bo.expandtab = true
    vim.bo.shiftwidth = 2
    vim.bo.softtabstop = 2
    vim.bo.tabstop = 2
    vim.bo.textwidth = 120
    vim.wo.number = true
    vim.wo.foldmethod = "syntax"
    -- Comment styling
    vim.cmd([[syntax match Namel3ssComment /^\s*#\s\zs\S.*$/ contains=Namel3ssCommentEmoji]])
    vim.cmd([[syntax match Namel3ssCommentEmoji /\v#\s\zs\S/ contained]])
    vim.api.nvim_set_hl(0, "Namel3ssComment", { fg = "#9CA3AF", italic = true })
    vim.api.nvim_set_hl(0, "Namel3ssCommentEmoji", { fg = "#f59e0b", italic = false })
  end,
})

-- Optional: Set up nvim-web-devicons for Namel3ss files
local ok, devicons = pcall(require, "nvim-web-devicons")
if ok then
  devicons.setup({
    override = {
      n3 = {
        icon = "",
        color = "#61afef",
        cterm_color = "74",
        name = "Namel3ss"
      },
      ai = {
        icon = "",
        color = "#61afef",
        cterm_color = "74",
        name = "Namel3ss"
      }
    },
    override_by_extension = {
      ["n3"] = {
        icon = "",
        color = "#61afef",
        name = "Namel3ss"
      },
      ["ai"] = {
        icon = "",
        color = "#61afef",
        name = "Namel3ss"
      }
    }
  })
end

return {}
