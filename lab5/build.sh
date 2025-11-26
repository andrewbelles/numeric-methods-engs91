#!/usr/bin/bash 

rm -f *.o ode quadrature *.csv *.png

gcc ode.c -o ode -lm 
g++ quadrature.cpp -o quadrature -lm 

