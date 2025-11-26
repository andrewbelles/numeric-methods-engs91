#!/usr/bin/bash 

summary_files=("$@")
png="all_convergence.png"

plot_cmd=""
for file in "${summary_files[@]}"; do 
  base="$(basename "$file")"
  label="${base%%_abs_error.csv}"
  label="${label//_/ }"

  clause="\"$file\" using 1:2 with linespoints title \"$label\""
  if [ -z "$plot_cmd" ]; then 
    plot_cmd="$clause"
  else 
    plot_cmd="$plot_cmd, \\
  $clause"
  fi 
done 

gnuplot <<EOF 
set datafile separator ","
set term pngcairo size 1200,800
set output "$png"

set xlabel "dt^{-1}"
set ylabel "absolute error at t = 2"
set title "Absolute error vs 1/dt (log-log comparison)"
set logscale x
set logscale y

plot \
  $plot_cmd

unset output
EOF

echo "wrote $png"

declare -A best_file
declare -A best_dt

shopt -s nullglob
for f in *_error_*.csv; do
  stem="${f%%_error_*}"
  dt="$(head -n 1 "$f")"
  # If we don't have a dt for this stem yet, or this dt is smaller, update.
  if [[ -z "${best_dt[$stem]}" ]]; then
    best_dt[$stem]="$dt"
    best_file[$stem]="$f"
  else
    # numeric compare (dt is floating)
    if awk -v a="${best_dt[$stem]}" -v b="$dt" 'BEGIN{exit !(b<a)}'; then
      best_dt[$stem]="$dt"
      best_file[$stem]="$f"
    fi
  fi
done
shopt -u nullglob

# Build gnuplot command
ts_png="methods_bestdt_timeseries.png"
ts_cmd=""
for stem in "${!best_file[@]}"; do
  f="${best_file[$stem]}"
  label="${stem//_/ }"
  clause="\"$f\" using 1:2 every ::1 with lines title \"$label\""
  if [ -z "$ts_cmd" ]; then
    ts_cmd="$clause"
  else
    ts_cmd="$ts_cmd, \\
  $clause"
  fi
done

# Only plot if we found any candidates
if [[ -n "$ts_cmd" ]]; then
  gnuplot <<EOF
set datafile separator ","
set term pngcairo size 1200,800
set output "$ts_png"

set xlabel "t"
set ylabel "relative error"
set title "Relative error over time (for smallest dt per method)"
set logscale y

plot \
  $ts_cmd

unset output
EOF

  echo "wrote $ts_png"
else
  echo "No *_error_*.csv files found to build finest-dt timeseries plot."
fi
