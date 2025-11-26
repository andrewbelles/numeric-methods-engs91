#!/bin/bash 
# 
# usage: 
#   ./run1.sh [int: fit type, 0 linear, 1 cubic, 2 linear on linearized]
#

if [ "$#" -ne 3 ]; then 
  echo "invalid usage: ./run.sh [lab4-data.txt] [fit type enum] [fit.png]"
  exit 1 
fi 

rm -f *.o approx *.png

g++ -std=c++20 -O3 -I"${GPINC:-.}" approx.cpp -o approx \
  -llapacke -llapack -lblas -lm

./approx "$1" "$2" "$3" 
xdg-open "$3"
