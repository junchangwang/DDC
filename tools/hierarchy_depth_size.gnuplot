############################## hierarchy_depth_size ##########################
# Per-bitmap compressed size vs density for L2 / L3 / L4 / L5 marker depths.
# X axis: density (= 1/cardinality), log scale, dense -> sparse (left -> right).
# Y axis: per-bitmap compressed size in MB, log scale (5-decade range).
# Style follows the eva/graphs_* gnuplot template (4 x 2.5 inch PDF, Linux
# Libertine O 25pt labels).
###############################################################################
reset

set terminal pdf size 4, 2.5 font 'Linux Libertine O,25'
set output "hierarchy_depth_size.pdf"

# Density goes from 50% (c=2, dense) down to 0.05% (c=2000, sparse).
# Invert so dense is on the LEFT, sparse on the RIGHT (matches motivation_eva).
set logscale x 10
set xrange [120:0.0006]
set xtics ("50%%" 50, "1%%" 1, "0.1%%" 0.1, "0.002%%" 0.002) font ',23' offset 0,0.5,0

set logscale y 10
set yrange [0.005:25]
set ytics ("0.01" 0.01, "0.1" 0.1, "1" 1, "10" 10) font ',25' offset 0.4,0

set mytics 10

set xlabel "Bit density" offset 0,1.25 font ',25'
set ylabel "Bitvector size (MB)" offset 1.35,0 font ',25'

set key font ',20' reverse Left right top at graph 1.00, 0.99 maxrows 2 width 2 samplen 2.0 spacing 1.0

set lmargin 6.6
set rmargin 0.2
set tmargin 0.1
set bmargin 1.75

set border lw 1.2

plot \
    "hierarchy_depth_size.dat" every 2 using 1:3 title "L1-L2" lc rgb "#A84141" lw 3 ps 1.5 pt 8  dt "-"   with linespoints, \
    "" every 2 using 1:4 title "L1-L3" lc rgb "#33807F" lw 3 ps 1.5 pt 12 dt 5     with linespoints, \
    "" every 2 using 1:5 title "L1-L4" lc rgb "#1F3FB0" lw 3 ps 1.5 pt 6           with linespoints, \
    "" every 2 using 1:6 title "L1-L5" lc rgb "#C57A21" lw 3 ps 1.5 pt 4  dt "."   with linespoints
