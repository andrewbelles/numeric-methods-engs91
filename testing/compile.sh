#!/usr/bin/bash 

rm -f *.o *.png ode
g++ -O3 adaptive_multistep.cpp -o ode 
