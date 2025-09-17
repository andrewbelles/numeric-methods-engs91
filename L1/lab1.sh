rm -f *.o *.png
rm recurrence 

g++ recurrence.cpp -lm -o recurrence 

cat bessel_front.txt | ./recurrence 1.0 15.0 40.0 1 | python plot.py --direction "Forward" 2>/dev/null

cat bessel_back.txt | ./recurrence 1.0 15.0 40.0 0 | python plot.py --direction "Backward" 2>/dev/null
