######################## hierarchy_depth_throughput_bar ########################
# OR / AND / COMP throughput by marker depth at c=100 (density 1%).
# X axis: L2 / L3 / L4 / L5 (4 groups).  Each group has 3 bars (OR, AND, COMP).
# Y axis: op/s (linear).
# L4 is the production default depth (OR/AND/COMP = motivation c=100 headline);
# L2/L3/L5 are depth-N variants.  Adding marker layers monotonically reduces
# throughput (one extra mask_expandloadu per region per layer), ~4-5% per layer
# and ~13% across L2->L5.  L4 sits between L3 and L5: it pays a small throughput
# cost over L3 but compresses ~3x better, justifying L4 as the production depth.
###############################################################################
reset

set terminal pdf size 4, 2.5 font 'Linux Libertine O,22'
set output "hierarchy_depth_throughput_bar.pdf"

set yrange [0:1280]
set ytics ("0" 0, "400" 400, "800" 800, "1200" 1200) font 'Linux Libertine O,18' offset 0.5,0

set xtics font 'Linux Libertine O,20' offset 0,0.3 scale 0

set xlabel "Hierarchy depth" offset 0,1.15 font 'Linux Libertine O,25'
set ylabel "Throughput (op/s)" offset 1.5,0 font 'Linux Libertine O,25'

set style data histogram
set style histogram cluster gap 1
set style fill pattern border lc rgb '#202020'
set boxwidth 0.85

# Horizontal legend along the top with uniform per-entry width so OR/AND/
# COMP labels (2/3/4 chars) don't bunch unevenly.
set key font 'Linux Libertine O,15' at graph 0.97, 0.99 \
    horizontal right top reverse Left samplen 1.2 spacing 1.0 width 1

set lmargin 6.4
set rmargin 0.2
set tmargin 0.1
set bmargin 1.95
set border lw 1.2
set tics nomirror

plot \
    "hierarchy_depth_throughput_bar.dat" using 2:xtic(1) title 'OR'   lc rgb '#A84141' fill pattern 7 border lc rgb '#A84141', \
    ""                                   using 3         title 'AND'  lc rgb '#33807F' fill pattern 6 border lc rgb '#33807F', \
    ""                                   using 4         title 'COMP' lc rgb '#643D83' fill pattern 2 border lc rgb '#643D83'
