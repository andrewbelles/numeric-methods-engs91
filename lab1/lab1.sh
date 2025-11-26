rm -f *.o *.png
rm -f recurrence 

g++ recurrence.cpp -lm -o recurrence 

cat ic_front.txt | ./recurrence 1.0 15.0 40.0 1 | python plot.py --direction "Forward" 

cat ic_back.txt | ./recurrence 1.0 15.0 40.0 0 | python plot.py --direction "Backward"
