rm -f *.o
rm recurrence 
clang++ recurrence.cpp -o recurrence 
cat initials.txt | ./recurrence 1.0 15.0 40.0 0 | python plot.py
