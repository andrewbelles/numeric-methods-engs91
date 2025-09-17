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
    parser.add_argument("--direction", required=True)
    args = parser.parse_args()

    line = sys.stdin.readline() # Read x values 
    vals = [float(x) for x in line.strip().split()]
    sys.stdin.readline()

    value_cnt = len(vals)
    bessel_list = []
    error_list  = []
     

    for _ in range(value_cnt):

        computed = [float(sys.stdin.readline()) for _ in range(51)]
        error    = [float(sys.stdin.readline()) for _ in range(51)]

        bessel_list.append(np.array(computed)) 
        error_list.append(np.abs(np.array(error)))
    
    for x, computed, error in zip(vals, bessel_list, error_list):
        f, ax = plt.subplots()
        n = np.linspace(0, 50, 51)
        ax.plot(n, computed)
        ax.set(title=f"x={x}, " + "Bessel $J_n(x)$, " + f"{args.direction}", xlabel="N", ylabel="$J_n(x)$")
        f.savefig(f"{x}_{args.direction}.png")
        plt.close(f)

        f, ax = plt.subplots()
        ax.semilogy(n, error)
        ax.set(title=f"x={x}, " + "Error $J_n(x) - \hat{J_n}(x)$, " + f"{args.direction}", xlabel="N", ylabel="$\log(\epsilon_n(x))$")
        f.savefig(f"{x}_{args.direction}_error.png")
        plt.close(f)

if __name__ == "__main__":
    main() 
