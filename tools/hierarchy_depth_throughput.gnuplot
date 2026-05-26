######################## hierarchy_depth_throughput ##########################
# COMP throughput (op/s) for ~((A|B) & (B|C)) vs density.
# L4 uses the production ComBit compressed-chain (operator| / and_no_bypass /
# operator~).  L2/L3/L5 use the depth-parameterised combit_n_or / combit_n_and
# / combit_n_not_inplace which share the same per-segment scratch + SIMD
# layer-compress fusion as production L4 — no decompress/recompress detour.
# X axis: density (= 1/cardinality), log scale, dense -> sparse (left -> right).
# Y axis: COMP throughput in op/s (linear, ~95-200 range).
###############################################################################
reset

set terminal pdf size 4, 2.5 font 'Linux Libertine O,25'
set output "hierarchy_depth_throughput.pdf"

set logscale x 10
set xrange [80:0.03]
set xtics ("50%%" 50, "10%%" 10, "1%%" 1, "0.1%%" 0.1) font ',23' offset 0,0.4,0

set yrange [80:230]
set ytics 30 font ',25'

set xlabel "Density" offset 0,1.2 font ',25'
set ylabel "COMP throughput (op/s)" offset 1.3,-0.9 font ',25'

set key font ',20' reverse Left left top at graph 0.04, 0.97 width -1 samplen 1.4 spacing 1.1 maxrows 2

set lmargin 7.0
set rmargin 0.4
set tmargin 0.3
set bmargin 1.9

set border lw 1.2

plot \
    "hierarchy_depth_throughput.dat" using 1:3 title "L2" lc rgb "#A84141" lw 3 ps 1.5 pt 8  dt "-"   with linespoints, \
    "" using 1:4 title "L3" lc rgb "#33807F" lw 3 ps 1.5 pt 12 dt 5     with linespoints, \
    "" using 1:5 title "L4" lc rgb "#1F3FB0" lw 3 ps 1.5 pt 6           with linespoints, \
    "" using 1:6 title "L5" lc rgb "#C57A21" lw 3 ps 1.5 pt 4  dt "."   with linespoints
