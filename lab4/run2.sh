#!/usr/bin/bash 

rm -f washer *.o washer_*.png

g++ -std=c++20 -O3 washer.cpp -o washer -llapacke -llapack -lopenblas -lm 

./washer 

xdg-open washer_angles.png
