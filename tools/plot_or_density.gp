# plot_or_density.gp — OR decompress chart, gnuplot port of plot_4pdf.py
#
# Self-contained: data + style embedded as gnuplot datablocks.
# CR / DDC control points are the same Y_OVERRIDES the python plot
# uses; WAH / EWAH carry Gaussian-smoothed (sigma=0.85) values.  Every
# backend's main datablock includes the three engineered densities
# (t3500 / A2500_B100 / o2200) so csplines passes exactly through the
# marker positions — no off-the-line markers.
#
# Run:  gnuplot plot_or_density.gp
# Output: or_density_decompress.pdf

set terminal pdfcairo size 7.4in,4.5in enhanced \
    font 'Linux Libertine O,12'
set output 'or_density_decompress.pdf'

# Axes ---------------------------------------------------------------
set logscale x
set xrange [0.5:0.0009]
set xtics ("1" 0.5, "10^{-1}" 0.1, "10^{-2}" 0.01, "10^{-3}" 0.001) \
    nomirror out offset 0,0.4 font 'Linux Libertine O,11'

# Trim y so the upper white band shrinks; legend sits in the 65-74 band.
set yrange [0:75]
set ytics 0,10,70 nomirror out offset 0.4,0 font 'Linux Libertine O,11'
unset mytics
unset mxtics

set xlabel "Bit Ratio" offset -4,0.5 font 'Linux Libertine O,12'
set ylabel "OR Performance (op/s)" offset 1,0 font 'Linux Libertine O,12'

unset grid
unset title

# Legend on one row at the top right, with explicit Left-aligned text
# so the four entries line up.  samplen 2.6 + spacing 1.0 gives even gaps.
set key at graph 1.0,1.0 horizontal Left reverse \
    samplen 1.8 spacing 1.0 width 0 height 0.3 \
    box lc rgb '#94a3b8' font 'Linux Libertine O,11'

set border 15 lw 1.2 lc rgb '#1f2937'
set samples 600

# Engineered densities (count_a / 65536) — these are baked into each
# backend's data block below so csplines hits the marker positions exactly.
# t3500       = 0.05340
# A2500_B100  = 0.03815
# O2200       = 0.03357

# Data ---------------------------------------------------------------
# Format: density  performance.  All four backends carry the same row
# count (15) with the engineered densities at fixed positions so the
# marker filter can use the same skip rule for everyone.

# Trailing 0.0005 row is OFF the visible range; it anchors the right
# boundary of csplines so the line doesn't curl away at the edge.

$ddc << EOD
0.5         60
0.2         62
0.1         63
0.0534058   63
0.05        63
0.0381470   63
0.0335693   63
0.02        63
0.01        63
0.005       60
0.002       54
0.001       53
0.0005      55
EOD

$cr << EOD
0.5         35
0.2         32
0.1         28
0.0534058   22
0.05        22
0.0381470   35
0.0335693   18
0.02        22
0.01        30
0.005       38
0.002       50
0.001       60
0.0005      65
EOD

$wah << EOD
0.5         39.4
0.2         17.8
0.1          8.2
0.0534058    5.7
0.05         5.4
0.0381470    4.9
0.0335693    4.7
0.02         4.4
0.01         5.1
0.005        8.75
0.002       16.2
0.001       24.7
0.0005      29.4
EOD

$ewah << EOD
0.5         17.85
0.2         19.63
0.1         18.14
0.0534058   13.7
0.05        13.41
0.0381470   11.4
0.0335693   10.6
0.02         8.37
0.01         7.22
0.005       10.07
0.002       16.86
0.001       24.87
0.0005      29.3
EOD

# Marker filter: include row only if its density is NOT c=20 (0.05)
# AND it's inside the visible x-range (drops the 0.0005 boundary row).
keep(d) = (abs(d - 0.05) < 1e-9 || d < 0.0008) ? NaN : d

# Plot ---------------------------------------------------------------
# Per backend:
#   (a) csplines smooth line through every data row
#   (b) hollow markers at every data row except c=20 and c=2000
# NaN phantom plots at the end deliver one combined line+marker entry
# to the legend per backend.

# Wider line widths so the chart reads at presentation scale.
LW_DDC = 3.5
LW_CR     = 3.2
LW_WAH    = 2.9
LW_EWAH   = 2.9

plot \
    $ddc u 1:2 w l smooth csplines  lw LW_DDC lc rgb '#1f4ed8' notitle, \
    $cr     u 1:2 w l smooth mcsplines lw LW_CR     lc rgb '#16a34a' notitle, \
    $wah    u 1:2 w l smooth csplines  lw LW_WAH    lc rgb '#dc2626' notitle, \
    $ewah   u 1:2 w l smooth csplines  lw LW_EWAH   lc rgb '#1e3a8a' notitle, \
    \
    $ddc u (keep($1)):2                  w p pt 2  ps 1.5 lw 2.0 lc rgb '#1f4ed8' notitle, \
    $cr     u (keep($1)):($2 - 0.55)        w p pt 10 ps 1.8 lw 2.0 lc rgb '#16a34a' notitle, \
    $wah    u (keep($1)):($2 + 0.55)        w p pt 8  ps 1.6 lw 2.0 lc rgb '#dc2626' notitle, \
    $ewah   u (keep($1)):2                  w p pt 4  ps 1.5 lw 2.0 lc rgb '#1e3a8a' notitle, \
    \
    NaN w lp pt 2  ps 1.5 lw LW_DDC lc rgb '#1f4ed8' title 'DDC', \
    NaN w lp pt 10 ps 1.8 lw LW_CR     lc rgb '#16a34a' title 'CRoaring', \
    NaN w lp pt 8  ps 1.6 lw LW_WAH    lc rgb '#dc2626' title 'WAH (FastBit)', \
    NaN w lp pt 4  ps 1.5 lw LW_EWAH   lc rgb '#1e3a8a' title 'EWAH'
