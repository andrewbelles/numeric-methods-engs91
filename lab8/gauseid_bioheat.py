#!/bin/python3.14  
# 
# gauss_hyperthermia.py Andrew Belles Nov 16th, 2025   
# 
# Gauss-Seidel Iteration w/ Relaxation to solve matrix system 
# from lab7 for FD Tridiagonal matrix 
# 

import numpy as np 
import matplotlib.pyplot as plt 
import numpy.typing as npt 


from typing import List, Dict, Optional   

EPS = np.float64(1e-12)
MAXITER: int = 1000

class Bioheat: 
    '''
    System Constants Intrinsic to the given Bioheat Equation 
    '''
    L       = np.float64(1.0)
    LAMBDSQ = np.sqrt(np.float64(2.7))
    SIG     = np.float64(100.0)
    GAMMA   = np.float64(-1.0 / L) 
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
        self.spectral_radii: Dict[int, npt.NDArray[np.float64]] = {}


    def sol(self):
        '''
        For all N in list nodes, generate a solution and key on the 
        the number of interior nodes 
        '''        
        for N in self.nodes: 
            print(f"{N} interior nodes...")
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
                a = np.sinh(np.sqrt(self.LAMBDSQ) * x)
                b = np.sinh(np.sqrt(self.LAMBDSQ) * self.L) 
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
        plt.axvline(21.0/self.L, ls=":", color="k", lw=0.7, label="N=20")
        plt.ylim([min(self.error - EPS), max(self.error)])
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
        return self.SIG * (h**2) * np.exp(self.GAMMA * (self.L - x))

    def matrix_solve_(self, system, b: npt.NDArray[np.float64]):
         return np.linalg.solve(system, b).astype(np.float64)

    def solve_(self, N: int, full: bool = False) -> npt.NDArray[np.float64]: 
        '''
        Solves the given Tridiagonal matrix from finite differences 
        '''
        h = (self.L) / (N + 1)
        a = -(2.0 + (self.LAMBDSQ) * (h**2))
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

class GaussBioheat(Bioheat): 
    
    def __init__(self, nodes: List[int]): 
        super().__init__(nodes, full=False)
        
    def spectral(self): 
        radii_vec = []
        for N in self.nodes: 
            radii_vec.append(self.spectral_radii[N])
        return np.array(radii_vec, dtype=np.float64).reshape((len(self.nodes),))
        

    def relax_(self, spectral_radius: np.float64) -> np.float64: 
        '''
        Optimal relaxation value given by Theorem 7.26, Burden and Faires 
        '''
        return 2.0 / (1.0 + np.sqrt(1 - (spectral_radius**2))) 

    def matrix_solve_(self, system: npt.NDArray[np.float64], 
                      b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: 
        '''
        Overrides Parent Method matrix_solve_ which just used np.linalg.solve
        Gauss-Seidel Iteration to solve distributed matrix system
        '''
        n = len(b) 
        z = np.zeros((n,), dtype=np.float64)  
        zprev = z.copy() 
        norm  = 0.0 
        nprev = 0.0
        spectral_radius = [] 

        for j in range(MAXITER):
            W = np.float64(1.0)
            for i in range(n): 
                aprev = system[i, :i]
                apost = system[i, i+1:]

                xprev = z[:i]
                xpost = z[i+1:]

                k = np.dot(aprev, xprev) + np.dot(apost, xpost)
                znext = (b[i] - k) / system[i, i]
                z[i] = W * znext + (1.0 - W) * z[i]

            # cauchy convergence < epsilon  
            nprev = norm 
            norm = np.linalg.norm(z - zprev, ord=np.inf)
            if norm <= EPS: 
                break 
           
            if j > 1: 
                spect = np.float64(norm / nprev) 
                if np.isfinite(spect) and spect < 1.0: 
                    spectral_radius.append(spect)  
                    W = self.relax_(spect)

            zprev[:] = z

        self.spectral_radii[n] = np.array(spectral_radius, dtype=np.float64).mean()
        return z 


def main(): 
    nodes = [5, 10, 20, 40, 80, 160, 320]
    model = GaussBioheat(nodes)
    model.analysis("Gauss-Seidel Iteration solution for Bioheat equation", 
                   "gauss-seidel-steadystate.png")
    spectral_radii = model.spectral()

    f = plt.figure(figsize=(12,12))
    plt.plot(nodes, np.array(spectral_radii), label="radii")
    plt.xticks(nodes)
    plt.xlim(0, 324)
    plt.xlabel("interior nodes [N]")
    plt.ylabel("estimated spectral radius")
    plt.title("Spectral Radius estimates as interior node count increases")
    plt.legend()
    f.savefig("spectral_radii.png")
    plt.close() 


if __name__ == "__main__": 
    main() 
