set nocompatible              " required for Vundle
filetype off                  " required
set modelines=0               " prevents some exploits
colorscheme molokai

" set the runtime path to include Vundle and initialize
" set rtp+=~/.vim/bundle/Vundle.vim
" call vundle#begin()

" Remapping kj to <Esc> for insert mode and some nice stuff
inoremap kj <Esc>
let mapleader = ","
nnoremap j gj
nnoremap k gk
nnoremap ; :
nnoremap <leader>W :%s/\s\+$//<cr>:let @/=''<CR>
nnoremap <leader>f :set foldmethod=indent<CR>
nnoremap <leader>g zR
nnoremap <leader>h zM
nnoremap <leader>v zO
nnoremap <leader>r zC
inoremap <leader>a <Esc>A
inoremap <leader>f <Backspace>
inoremap <leader>c <End><Space><Space>#<Space>
nnoremap <leader>c A<Space><Space>#<Space>
inoremap <leader>C <End><Space><Space>//<Space>
nnoremap <leader>C A<Space><Space>//<Space>
nnoremap <leader>xx :%!xxd<CR>
nnoremap <leader>xy :%!xxd -r<CR>

" Working with windows
nnoremap <leader>w <C-w>v<C-w>l
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Auto closing
inoremap " ""<left>
"inoremap ' ''<left>
inoremap ( ()<left>
inoremap [ []<left>
inoremap { {}<left>
inoremap {<CR> {<CR>}<ESC>O
inoremap {;<CR> {<CR>};<ESC>O

" Tab stuff
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab

" Home sweet home
set encoding=utf-8
set scrolloff=3
set autoindent
set showmode
set showcmd
set hidden
set wildmenu
set wildmode=list:longest
set visualbell
set cursorline
set ttyfast
set ruler
set backspace=indent,eol,start
set laststatus=2
set relativenumber
set number
" set undofile

" Search stuff
nnoremap / /\v
vnoremap / /\v
set ignorecase
set smartcase
set gdefault
set incsearch
set showmatch
set hlsearch
nnoremap <leader><space> :noh<cr>

" Moving code easier
vnoremap < <gv
vnoremap > >gv

" Fix for end of lines
" set wrap
" set textwidth=79
" set formatoptions=qrn1
" set colorcolumn=80

"Plugin 'file:///home/dandy/.vim/bundle/YouCompleteMe'
"let g:ycm_global_ycm_extra_conf = '~/.vim/.ycm_extra_conf.py'

filetype plugin indent on    " required
