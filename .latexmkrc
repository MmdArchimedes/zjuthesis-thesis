$out_dir="out";
$pdf_mode=5;
# 根目录若残留与 out/ 同名的 zjuthesis.bcf / .bbl，biber 可能优先读到旧控制文件，导致 .bbl 与正文脱节（大量 Citation undefined）。
BEGIN {
  for my $f (qw(zjuthesis.bcf zjuthesis.bbl zjuthesis.blg)) {
    unlink($f) if (-e $f && -e "out/$f");
  }
}
# latexmk 默认「biber %S」中 %S 为 out/zjuthesis 时，易触发 kpsewhich 与路径问题。改为在 aux 目录下读写，且用 %R（主文件名、无路径）作为 jobname。
$biber = 'biber --input-directory=%V --output-directory=%V %O %R';
$xelatex="xelatex -synctex=1";
$xdvipdfmx="xdvipdfmx -q -E -o %D %O %S";
$clean_ext = 'thm bbl hd loe xdv run.xml nlg nls';
$makeindex = 'makeindex -s gind.ist %O -o %D %S';

# Custom dependency and function for nomencl package 
add_cus_dep( 'nlo', 'nls', 0, 'makenlo2nls' );
sub makenlo2nls {
 system("makeindex \"$_[0].nlo\" -s nomencl.ist -o \"$_[0].nls\" -t \"$_[0].nlg\"" );
}

@default_files=('zjuthesis.tex')
