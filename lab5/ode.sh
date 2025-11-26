#!/usr/bin/bash 

./build.sh 
./ode 

./plot.sh "Euler" euler_traj_*.csv  
./plot.sh "Midpoint" midpoint_traj_*.csv  
./plot.sh "Modified Euler" mod_euler_traj_*.csv
./plot.sh "AB-AM PCS" abam_traj_*.csv
./plot.sh "Runge-Kutta 4th Order" rk4_traj_*.csv

./error.sh "Euler" euler euler_error_*.csv 
./error.sh "Midpoint" midpoint midpoint_error_*.csv  
./error.sh "Modified Euler" mod_euler mod_euler_error_*.csv
./error.sh "AB-AM PCS" abam abam_error_*.csv
./error.sh "Runge-Kutta 4th Order" rk4 rk4_error_*.csv

./convg.sh euler_abs_error.csv midpoint_abs_error.csv mod_euler_abs_error.csv \
  abam_abs_error.csv rk4_abs_error.csv 

rm -f *.csv
