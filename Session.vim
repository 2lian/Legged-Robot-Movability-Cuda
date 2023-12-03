let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Legged-Robot-Movability-Cuda
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 ~/Legged-Robot-Movability-Cuda
badd +6 cpp_code.cpp
badd +2 cuda_code.cu
badd +1 one_leg.cu
badd +1 static_variables.cpp
badd +10 Header.h
badd +1 HeaderCUDA.h
badd +2 HeaderCPP.h
badd +1 Makefile
badd +20 CMakeLists.txt
badd +1 externals/eigen-3.4.0/INSTALL
badd +3 math_util.cpp
badd +1 LAUNCH.bash
badd +17 compile_commands.json
badd +1 externals/eigen-3.4.0/Eigen/Dense
badd +39 term://~/Legged-Robot-Movability-Cuda//4934:.\ LAUNCH.bash
badd +4336 ~/.local/state/nvim/lsp.log
badd +8 .clangd
badd +1 /usr/include/c++/9/iostream
badd +823 externals/tinycolormap-master/include/tinycolormap.hpp
badd +0 fugitive:///home/elian/Legged-Robot-Movability-Cuda/.git//
argglobal
%argdel
$argadd ~/Legged-Robot-Movability-Cuda
edit fugitive:///home/elian/Legged-Robot-Movability-Cuda/.git//
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 22 + 23) / 46)
exe '2resize ' . ((&lines * 21 + 23) / 46)
argglobal
balt ~/Legged-Robot-Movability-Cuda
setlocal fdm=manual
setlocal fde=0
setlocal fmr=<<<<<<<<,>>>>>>>>
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 14 - ((13 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 14
normal! 012|
lcd ~/Legged-Robot-Movability-Cuda
wincmd w
argglobal
if bufexists(fnamemodify("~/Legged-Robot-Movability-Cuda/LAUNCH.bash", ":p")) | buffer ~/Legged-Robot-Movability-Cuda/LAUNCH.bash | else | edit ~/Legged-Robot-Movability-Cuda/LAUNCH.bash | endif
if &buftype ==# 'terminal'
  silent file ~/Legged-Robot-Movability-Cuda/LAUNCH.bash
endif
balt ~/Legged-Robot-Movability-Cuda
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 5 - ((4 * winheight(0) + 10) / 21)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 5
normal! 06|
lcd ~/Legged-Robot-Movability-Cuda
wincmd w
exe '1resize ' . ((&lines * 22 + 23) / 46)
exe '2resize ' . ((&lines * 21 + 23) / 46)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
