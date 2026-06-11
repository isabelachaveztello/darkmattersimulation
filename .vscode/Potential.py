"""
This file contains the Potential class which is used to compute the trapping
potential of the ions. The potential is a sum of the coloumb potential and a
polynomial trapping potential. The polynomial trapping potential is defined by a
set of coefficients and the coloumb potential is defined by the positions of the
ions. The class also contains methods to compute the hessian of the trapping
potential and the eigenvalues and eigenvectors of the hessian.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize
from autograd import hessian as ag_hessian

class MockConstants:
    AMU = 1.66053907e-27  # converts AMU to kg
    K = 8.9875e9          # Coulomb's constant
    CHARGE = 1.62176634e-19 # Elementary charge
    OMEGA_0 = 36* 2*math.pi*10**6  # placeholder for length_scale calculation (HZ?)

constants = MockConstants()

class Potential:
    potential_coeff_matrix: np.ndarray
    length_scale: float
    hess_tuple: tuple
    eq_positions: np.ndarray

    def __init__(self, n_ion: int, mass: float, order: int, fsec: float, dimensionality: int = 3):
        self.order = order
        self.n_ion = n_ion
        self.m = mass * constants.AMU
        self.fsec = fsec
        self.d = dimensionality

        self.potential_coeff_matrix = np.zeros((self.d, self.order))
        self.length_scale = 1
        self.hess_tuple = None
        self.eq_positions = None

    def _coulomb_potential(self, a: np.ndarray) -> float:
        """
        Computes the Coloumb Potential
        """
        csum = 0
        x = a.reshape(self.d, self.n_ion)
        for i in range(self.n_ion):
            for j in range(i):
                den = sum((x[k][i] - x[k][j]) ** 2 for k in range(self.d))
                csum += 1 / math.sqrt(den)
        return csum

    def _trap_potential(self, x: np.ndarray) -> float:
        """
        Computes a polynomial trapping potential for a given position given a
        coefficient expansion of a trapping potential
        """
        tsum = 0
        cf_flat = self.potential_coeff_matrix.flatten()
        for i in range(self.n_ion):
            for j in range(1, self.order + 1):
                for k in range(self.d):
                    tsum += x[k * self.n_ion + i] ** j / math.factorial(j) * cf_flat[k * self.order + (j - 1)]
        return tsum
    
    def set_coeffs(self, coeff_matrix: np.ndarray):
        """
        Sets the coefficients of the polynomial trapping potential
        """

        self.potential_coeff_matrix = coeff_matrix
        if self.d == 3:
            rf = (
                (self.fsec * 1e6 * 2 * math.pi) ** 2
                * self.m
                * self.length_scale**3
                / constants.K
                / (constants.CHARGE**2)
            )
            self.potential_coeff_matrix[1][1] = rf
            self.potential_coeff_matrix[2][1] = rf
            print(f'Your RF is {rf}')
    
    def set_length_scale(self, omega_0: float = constants.OMEGA_0):
        """
        Sets the relative length scale for the system
        """
        self.length_scale = (constants.K * constants.CHARGE**2 / (self.m * (omega_0 * 2 * math.pi) ** 2)) ** (1 / 3)

    def get_pos(self) -> np.ndarray:
        """
        Returns equilibrium positions of the ions in meters
        """

        return np.array(self.eq_positions * self.length_scale)

    def get_mode(self) -> tuple:
        """
        Returns the full list of eigenvalues, eigenvectors and hessian of the trapping potential
        """

        return self.hess_tuple

    def total_potential_energy(self, x: np.ndarray) -> float:
        """
        Computes the total trapping potential which is a summation of the coloumn and polynomial
        """

        # self.set_length_scale()
        return self._coulomb_potential(x) + self._trap_potential(x)

    def poly_potential_energy(self, x: np.ndarray) -> float:
        return self._trap_potential(x)

    def equilibrium_pos(self):
        """
        Finds the point of mininum trapping potential in unitless parameters
        """

        k = np.arange(-self.n_ion / 2, self.n_ion / 2, 1)

        restore3 = False
        old_c = None

        if self.d == 3:
            restore3 = True
            old_c = self.potential_coeff_matrix

        self.potential_coeff_matrix = np.reshape(
            self.potential_coeff_matrix[0], (1, len(self.potential_coeff_matrix[0]))
        )
        self.d = 1

        res = scipy_minimize(
            self.total_potential_energy, k, method='nelder-mead', options={'xatol': 1e-15, 'disp': False}
        )
        eq = np.reshape(res.x, (self.d, self.n_ion))

        if restore3:
            self.d = 3
            self.potential_coeff_matrix = old_c
            r = np.zeros((self.d, self.n_ion))
            r[0] = np.sort(eq[0])
            r[1] = [0] * self.n_ion
            r[2] = r[1]
            eq = r
        self.eq_positions = eq
    
    def calc_equidistant_potential(self):
        pot_guess = .0001*np.random.rand(2)
        f = lambda x: self.eq_dist_loss(x)
        res = scipy_minimize(f,pot_guess,method='nelder-mead',options={'xatol': 1e-15, 'disp': False},bounds=((0,None),(0,None)))
        potential_coeff_matrix = np.zeros((self.d,self.order))
        potential_coeff_matrix[0]=np.array([0,res.x[0],0,res.x[1]])
        self.set_coeffs(potential_coeff_matrix)
        self.equilibrium_pos()
    
    def eq_dist_loss(self,pot):
        potential_coeff_matrix = np.zeros((self.d,self.order))
        potential_coeff_matrix[0]=np.array([0,pot[0],0,pot[1]])
        self.set_coeffs(potential_coeff_matrix)
        self.equilibrium_pos()
        avg_dist = 0
        for i in range(len(self.eq_positions[0])-1):
            avg_dist+=np.abs(self.eq_positions[0][i+1]-self.eq_positions[0][i])
        avg_dist/=(self.n_ion-1)
        mean_error = 0
        for i in range(len(self.eq_positions[0])-1):
            mean_error+=(avg_dist-np.abs(self.eq_positions[0][i+1]-self.eq_positions[0][i]))**2
        return mean_error
    
    def analyze_pos(self):
        dist=[]
        for i in range(len(self.eq_positions[0])-1):
            dist.append(np.abs(self.eq_positions[0][i+1]-self.eq_positions[0][i]).item())
        return dist

    def hess_coulomb(self) -> NDArray[np.float64]:
        """
        Analytically solves for the hessian of the coulomb potential.
        """

        n_total = self.d * self.n_ion
        hess_c = np.zeros((n_total, n_total))
        for i in range(n_total):
            diff = np.subtract(self.eq_positions[:, i % self.n_ion].reshape(self.d, 1), self.eq_positions)
            rsq = np.sum(diff**2, axis=0)
            r = rsq**0.5
            diag = 3 * diff**2 - r**2
            not_diag = r**2 - 3 * diff**2
            r[i % self.n_ion] = 10e10  # Temporary handling of divide-by-zero
            for j in range(n_total):
                if i == j:
                    hess_c[i][j] = np.sum(np.divide(diag, r ** (5)), axis=1)[i // self.n_ion]
                elif i // self.n_ion == j // self.n_ion:
                    hess_c[i][j] = np.divide(not_diag, r ** (5))[i // self.n_ion][j % self.n_ion]
                else:
                    hess_c[i][j] = 0
        return hess_c

    def calc_hessian(self) -> None:
        """
        Computes the full hessian of the trapping potential alongside its eigenvectors and eigenvalues
        """

        hessian_fn = ag_hessian(self.poly_potential_energy)
        hessian = hessian_fn(self.eq_positions.flatten()) + self.hess_coulomb()
        hessian *= (constants.CHARGE**2 * constants.K) / (self.m * self.length_scale**3)

        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        eigenvalues = np.sqrt(eigenvalues) / (2 * math.pi)
        self.hess_tuple = (hessian, eigenvalues, eigenvectors)

    def transverse_normal_modes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the eigenvalues and eigenmodes of only the transverse normal modes
        """

        _, eigenvalues, eigenvectors = self.get_mode()
        eigenvectors = np.real(eigenvectors)
        eigenvalues = np.real(eigenvalues)

        sign = np.sign(eigenvectors[self.n_ion])[self.n_ion : 2 * self.n_ion]

        indices = np.argsort(eigenvalues[self.n_ion : 2 * self.n_ion])[::-1]

        freqs = eigenvalues[self.n_ion : 2 * self.n_ion][indices]
        modes = eigenvectors[self.n_ion : 2 * self.n_ion, self.n_ion : 2 * self.n_ion] / sign
        modes = modes[:, indices]
        return freqs, modes