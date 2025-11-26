#!/bin/python3.14 

from typing import List, Dict, Optional  
import numpy.typing as npt 

import numpy as np 
import matplotlib.pyplot as plt 

class Bioheat: 
    '''
    System Constants Intrinsic to the given Bioheat Equation 
    '''
    L       = np.float64(1.0)
    LAMBDA  = np.sqrt(np.float64(2.7))
    SIG     = np.float64(100.0)
    GAMMA   = -L 
    CORE    = np.float64(37.0)
    SURFACE = np.float64(32.0)
    ARTERY  = CORE
    BCLEFT  = np.float64(CORE - ARTERY)
    BCRIGHT = np.float64(SURFACE - ARTERY)
    
    def __init__(self, nodes: List[int], full: bool = False): 
        '''
        nodes: 
            Could be a list of N interior nodes to solve for
        '''
        self.nodes: List[int] = nodes 
        self.solutions: Dict[int, npt.NDArray[np.float64]] = {}
        self.truth: Dict[int, npt.NDArray[np.float64]] = {}
        self.error: npt.NDArray[np.float64] = np.zeros((len(nodes),)) 
        self.full = full 


    def sol(self):
        '''
        For all N in list nodes, generate a solution and key on the 
        the number of interior nodes 
        '''        
        for N in self.nodes: 
            self.solutions[N] = self.solve_(N, self.full)


    def compute_analytic_(self): 
        '''
        Using smallest time step, compute analytic solution 
        We will use this solution to compare against all solutions 
        '''

        for N in self.nodes: 
            self.truth[N] = np.zeros((N,), dtype=np.float64)

            h = (self.L) / np.float64(N + 1)

            for i in range(N):
                x = np.float64(np.float64(i + 1) * h)
                a = np.sinh(self.LAMBDA * x)
                b = np.sinh(self.LAMBDA * self.L) 
                delta = self.SURFACE - self.CORE
                self.truth[N][i] = delta * (a / b)

    
    def best(self):
        return self.solutions[self.nodes[-1]]

    def analytic(self):
        return self.truth[self.nodes[-1]]


    def analysis(self, title: str, png: str):
        ''' 
        Computes truth dict, then finds maximum difference from analytic solution 
        compared to computed solution. Plots 1/dx vs error on log-log scale 
        '''
        if not self.truth:   
            self.compute_analytic_()

        if not self.solutions: 
            self.sol()

        inv_step = np.zeros((len(self.nodes),))

        for (i, N) in enumerate(self.nodes):
            res   = self.solutions[N]
            truth = self.truth[N] 

            eps = np.max(np.abs(res - truth))

            self.error[i] = eps
            h = self.L / (N + 1) 
            inv_step[i]   = 1.0 / h 

        p, _ = np.polyfit(np.log(inv_step), np.log(self.error), 1)
        print(f"order={p}")

        f = plt.figure(figsize=(12,7))
        plt.loglog(inv_step, self.error, label="error")
        plt.xlabel(R"1/$\Delta x$  [log-scale]")
        plt.ylabel(R"$\max|\tilde{T}_i - \tilde{T}_i^a|$  [log-scale]")
        plt.title(title)
        plt.legend()
        f.savefig(png)
        plt.close() 

    def plot(self, title: str, png: str, 
             against: Optional[List[npt.NDArray[np.float64]]]=None, 
             ag_labels: Optional[List[str]]=None): 
        if not self.solutions: 
            self.sol() 

        N = self.nodes[-1]
        h = (self.L) / np.float64(N + 1)
        T = self.solutions[N]
        x = h * np.arange(1, N + 1, dtype=np.float64)

        f = plt.figure(figsize=(12,7))
        plt.plot(x, T, label="inhomogeneous")

        if against is not None and ag_labels is not None: 
            for (ya, aglab) in zip(against, ag_labels):
                plt.plot(x, ya, label=aglab)
        plt.xlabel(R"$x$ [$m$]")
        plt.ylabel(R"temperature [$C^{\circ}$]")
        plt.title(title)
        plt.legend() 
        f.savefig(png)
        plt.close() 


    def exp_(self, x: npt.NDArray[np.float64], h: np.float64):
        return -self.SIG * (h**2) * np.exp(-self.GAMMA * (self.L - x))

    def matrix_solve_(self, system, b: npt.NDArray[np.float64]):
         return np.linalg.solve(system, b).astype(np.float64)

    def solve_(self, N: int, full: bool = False) -> npt.NDArray[np.float64]: 
        '''
        Solves the given Tridiagonal matrix from finite differences 
        '''
        h = (self.L) / (N + 1)
        a = -(2.0 + (self.LAMBDA**2) * (h**2))
        main = np.full(N, a, dtype=np.float64)
        off  = np.ones(N - 1, dtype=np.float64)

        system = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

        b = np.zeros(N, dtype=np.float64) 
        if full: 
            xs = h * np.arange(1, N + 1, dtype=np.float64)
            b -= self.exp_(xs, h) 

        b[0]  -= self.BCLEFT
        b[-1] -= self.BCRIGHT  

        return self.matrix_solve_(system, b)

def main(): 
    homogeneous   = Bioheat([5, 10, 20, 40, 80, 160, 320, 640, 1280])
    inhomogeneous = Bioheat([5, 10, 20, 40, 80, 160, 320, 640, 1280], full=True)

    homogeneous.analysis(
        "Error of Finite Difference Method for Steady-State Equation",
        "error_homogeneous.png"
    )

    inhomogeneous.analysis(
        "Error of Finite Difference Method for Steady-State Equation",
        "error_inhomogeneous.png"
    )

    hom_best = homogeneous.best()  

    inhomogeneous.plot(
        "Steady-State solutions of Homogeneous and Inhomogeneous Equations",
        "bioheat_steadystate.png",
        [hom_best], 
        ["homogeneous"]
    )

if __name__ == "__main__":
    main()
