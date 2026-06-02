###################### hierarchy_depth_runlength_bar #########################
# COMP throughput by marker depth at c=500 on run-length clustered bitmaps.
# X axis: L2 / L3 / L4 / L5  (4 categorical bars).
# Y axis: COMP op/s (linear).
# Data shape: column store sorted by the indexed attribute (TPC-H Q6 /
# sorted shipdate).  L4's 32K-bit batch-skip fires on ~99% of batches,
# matching its design target — c=500 chosen as a typical post-filter
# selectivity for OLAP range predicates on sorted timeseries columns
# (a one-month window in TPC-H lineitem shipdate is in this density
# range).
###############################################################################
reset

set terminal pdf size 4, 2.5 font 'Linux Libertine O,25'
set output "hierarchy_depth_runlength_bar.pdf"

set xrange [-0.7:3.7]
set yrange [0:1900]
set ytics 400 font ',25' offset 0.5,0

set xtics font ',25' offset 0,0.4 scale 0

set xlabel "Marker depth" offset 0,1.4 font ',25'
set ylabel "COMP throughput (op/s)" offset 1.6,-0.9 font ',25'

set style data histogram
set style histogram cluster gap 1
set style fill solid 1.0 border lc rgb '#202020'
set boxwidth 0.65

set lmargin 7.0
set rmargin 0.4
set tmargin 0.6
set bmargin 1.9
set border lw 1.2
set tics nomirror

unset key

plot \
    "hierarchy_depth_runlength_bar.dat" using 2:xtic(1) lc rgb '#A84141' notitle, \
    "" using ($0):($2 + 70):(sprintf('%.0f', $2)) with labels font 'Linux Libertine O,18' textcolor rgb '#202020' notitle
