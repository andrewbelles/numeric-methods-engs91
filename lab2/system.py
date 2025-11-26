#
# system.py  Andrew Belles  Sept 30th, 2025 
# 
# System of nonlinear equations solver   
# 
# 
# 

import numpy as np 
import matplotlib.pyplot as plt 

r1 = np.float64(43.0) 
r2 = np.float64(23.0)
r3 = np.float64(33.0) 
r4 = np.float64(9.0) 
t4 = np.float64(1.13446) # 65 deg in rads 

EPS = np.float64(1e-9)
MAX_ITER = 100 

def f1(x2, x3): 
    return r2 * np.cos(x2) + r3 * np.cos(x3) + r4 * np.cos(t4) - r1 

def f2(x2, x3): 
    return r2 * np.sin(x2) + r3 * np.sin(x3) + r4 * np.sin(t4) 

def system(x): 
    s = np.array([[r2 * np.cos(x[0]) + r3 * np.cos(x[1]) + r4 * np.cos(t4) - r1],
                     [r2 * np.sin(x[0]) + r3 * np.sin(x[1]) + r4 * np.sin(t4)]])
    return s.reshape(-1) 


# System jacobian at \vec{x} 
def jacobian(x):
    return np.array([[-r2 * np.sin(x[0]), -r3 * np.sin(x[1])],
                     [r2 * np.cos(x[0]), r3 * np.cos(x[1])]])


# Single step of newtons 
def newton(x): 
    '''
    input:  x  := (2,)
    output: fx := (2,)
    '''
    # xk - J^{-1}(xk) * F(xk)
    return x - ( np.linalg.solve(jacobian(x), system(x)) ) 


def main(): 

    # Plot region that we are interested in 
    vec_x2 = np.linspace(0, np.pi, 1000)
    vec_x3 = np.linspace(0, 2.0 * np.pi, 1000)
    X2, X3 = np.meshgrid(vec_x2, vec_x3)

    F1 = f1(X2, X3) 
    F2 = f2(X2, X3)

    f, ax = plt.subplots()
    c1 = ax.contour(X2, X3, F1, levels=[0.0], colors="C0", linewidths=2)
    c2 = ax.contour(X2, X3, F2, levels=[0.0], colors="C1", linewidths=2)

    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()

    ax.set_xlabel(r"$\theta_2$")
    ax.set_ylabel(r"$\theta_3$")
    v = ax.axvline(x=1, linestyle="--", linewidth=1)
    h = ax.axhline(y=5, linestyle="--", linewidth=1)
    ax.set_title("Initial Values for Nonlinear Newton's")
    plt.tight_layout()
    ax.legend(
        handles=[h1[0], h2[0], v, h],
        labels=[r"$f_1(\theta_2,\theta_3)=0$", r"$f_2(\theta_2,\theta_3)=0$",
        r"$\theta_{2,0}=1$", r"$\theta_{3,0}$=5"]
    )
    f.savefig("approximate_nonlinear.png")
    plt.close() 

    # Sensible bounds qualitatively determined qualitatively from graph of eq-ns  
    x2 = 1
    x3 = 5

    x = np.array([[x2],[x3]]).reshape(-1)
    error = []
    cerr  = np.abs(np.linalg.norm(system(x))) 

    i: int = 0 
    while np.abs(cerr) > EPS and i < MAX_ITER:
        
        x = newton(x)
        cerr = np.abs(np.linalg.norm(system(x)))
        error.append(cerr)
        i += 1  

    f = plt.figure()
    plt.semilogy(np.arange(1, len(error)+1), error, label="log error")
    plt.title(f"root={x}") 
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error [log-scale]")
    plt.legend() 
    f.savefig("system_error.png")

    # Plot solution on region of interest  
    f, ax = plt.subplots()
    c1 = ax.contour(X2, X3, F1, levels=[0.0], colors="C0", linewidths=2)
    c2 = ax.contour(X2, X3, F2, levels=[0.0], colors="C1", linewidths=2)

    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()

    ax.set_xlabel(r"$\theta_2$")
    ax.set_ylabel(r"$\theta_3$")
    v = ax.axvline(x=x[0], linestyle="--", linewidth=1)
    h = ax.axhline(y=x[1], linestyle="--", linewidth=1)
    ax.set_title("Initial Values for Nonlinear Newton's")
    plt.tight_layout()
    ax.legend(
        handles=[h1[0], h2[0], v, h],
        labels=[r"$f_1(\theta_2,\theta_3)=0$", r"$f_2(\theta_2,\theta_3)=0$",
                r"$\theta_{2}^*=$"+f"{x[0]:.5f}", r"$\theta_{3}^*$="+f"{x[1]:.5f}"]
    )
    f.savefig("solution_nonlinear.png")
    plt.close() 
    
if __name__ == "__main__":
    main()
