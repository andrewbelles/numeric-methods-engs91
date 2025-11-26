#!/usr/bin/bash 

# get method name 
method="$1"
shift 

# for file on cli args, process under assumption it was with specified method 
for file in "$@"; do 
  out="${file%.csv}.png"
  step=$(head -n 1 "$file")
  
  gnuplot <<EOF
set datafile separator ","
set term pngcairo size 1200,800
set output "$out"

set xlabel "t"
set ylabel "y(t)"
set title "$method vs Analytic: (dt = $step)"

# 'every ::1' skips the first line (row 0), which is the stepsize metadata
plot \
  "$file" using 1:2 every ::1 with lines title "$method approx", \
  "$file" using 1:3 every ::1 with lines title "Analytic"

unset output
EOF
  
  echo "wrote $out"

done 
