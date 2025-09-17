# 
# plot.py  Andrew Belles  Sept 16th, 2025 
# 
# Plot tool, reads numeric values from stdin
# plots corresponding graph 
# 
# 

import argparse, sys  
import matplotlib.pyplot as plt  
import numpy as np 

def main(): 
    '''
    Read input values from stdin, get direction from argparse 
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--iter", default=50, type=int)

    args = parser.parse_args()
    line = sys.stdin.readline() # Read x values 
    vals = [float(x) for x in line.strip().split()]

    value_cnt = len(vals)
    bessel_list = []

    for _ in range(value_cnt):
    
        bessel_vals = []
        for _ in range(args.iter):
            line = sys.stdin.readline()
            line = line.strip()
            if not line: 
                continue
            
            bessel_vals.append(float(line))

        bessel_list.append(np.array(bessel_vals)) 

    f, ax = plt.subplots()
    n = np.linspace(0, 50, 50) 
    for x, y in zip(vals, bessel_list):
        ax.plot(n, y, label=f"x={x:g}")

    ax.set(title="Bessel J_n(x)", xlabel="nth", ylabel="J_n(x)")
    ax.legend()
    f.tight_layout()

    plt.show()

if __name__ == "__main__":
    main() 
