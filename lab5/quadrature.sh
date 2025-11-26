#!/usr/bin/bash 

rm -f *.o quadrature 

g++ quadrature.cpp -o quadrature -lm 

./quadrature 
