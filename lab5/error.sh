#!/usr/bin/bash 

method="$1"
out="$2"
shift 

error_timeseries=()

# organize read files 

for file in "$@"; do 
  case "$file" in 
    *_error_*.csv)
      error_timeseries+=("$file")
      ;;
    *)
      ;; 
  esac 
done 

dts=()
for file in "${error_timeseries[@]}"; do 

  dt=$(head -n 1 "$file")
  dts+=( "$dt" )

done 

timeseries="${out}_error_all.png"
plot_cmd=""
for idx in "${!error_timeseries[@]}"; do
  f="${error_timeseries[$idx]}"
  dt_label="${dts[$idx]}"

  # one clause of the form:
  #   "file" using 1:2 every ::1 with lines title "dt=..."
  clause="\"$f\" using 1:2 every ::1 with lines title \"dt = $dt_label\""

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
set output "$timeseries"

set xlabel "t"
set ylabel "relative error"
set title "$method relative error over time"
set logscale y

plot \
  $plot_cmd

unset output
EOF

echo "wrote $timeseries"
