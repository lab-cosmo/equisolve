from typing import Union

import metatensor
import numpy as np
import scipy
from metatensor import Labels, TensorBlock, TensorMap
from metatensor.operations import dot, multiply, ones_like, slice

from ..utils import block_to_array, dict_to_tensor_map, tensor_map_to_dict


def compute_sparse_kernel(
    tensor: TensorMap, pseudo_points: TensorMap, degree: int
) -> TensorMap:
    metatensor.pow(metatensor.dot(tensor, pseudo_points), degree)


class SparseKernelRidge:
    def __init__(
        self,
        parameter_keys: Union[List[str], str] = None,
    ) -> None:
        if type(parameter_keys) not in (list, tuple, np.ndarray):
            self.parameter_keys = [parameter_keys]
        else:
            self.parameter_keys = parameter_keys

        self._weights = None

    def fit(
        self,
        k_mm: TensorMap,
        k_nm: TensorMap,
        k_y: TensorMap,
        alpha: Union[float, TensorMap] = 1.0,
        jitter: float = 1e-13,
        solver: str = "auto",
        cond: float = None,
    ):
        if type(alpha) is float:
            alpha_tensor = ones_like(k_y)

            samples = Labels(
                names=k_y.sample_names,
                values=np.zeros([1, len(k_y.sample_names)], dtype=int),
            )

            alpha_tensor = slice(alpha_tensor, samples=samples)
            alpha = multiply(alpha_tensor, alpha)
        elif type(alpha) is not TensorMap:
            raise ValueError("alpha must either be a float or a TensorMap")

        for key, yblock in k_y:
            k_nm_block = k_nm.block(key)
            k_mm_block = k_mm.block(key)
            structures = np.unique(k_nm_block.samples["structure"])
            n_atoms_per_structure = []
            for structure in structures:
                n_atoms = np.sum(k_nm_block.samples["structure"] == structure)
                n_atoms_per_structure.append(float(n_atoms))

            delta = np.std(yblock.values)
            energy_regularizer = np.sqrt(n_atoms_per_structure)
            # yblock.values[:] /= energy_regularizer

            _alpha = block_to_array(
                alpha.block(key), parameter_keys=self.parameter_keys
            )
            Y = block_to_array(yblock, parameter_keys=self.parameter_keys)
            n_centers = yblock.value.shape[0]
            Y[:n_centers] /= energy_regularizer[:, None]
            Y /= _alpha / delta

            KNM = block_to_array(k_nm_block, parameter_keys=self.parameter_keys)
            KNM[:n_centers] /= energy_regularizer[:, None]
            KNM /= _alpha / delta

            KMM = block_to_array(k_mm_block, parameter_keys=self.parameter_keys)

    # check that alpha has the the same samples as k_nm and only one property

    def predict(self):
        pass


class SparseGPRSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the sparse GPR problem::

        b = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM

    Parameters
    ----------
    KMM : numpy.ndarray
        KNM matrix
    regularizer : float
        regularizer
    jitter : float
        numerical jitter to stabilize fit
    solver : {'RKHS-QR', 'RKHS', 'QR', 'solve', 'lstsq'}
        Method to solve the sparse KRR equations.

        * RKHS-QR: TBD
        * RKHS: Compute first the reproducing kernel features by diagonalizing K_MM
          and computing `P_NM = K_NM @ U_MM @ Lam_MM^(-1.2)` and then solves the linear
          problem for those (which is usually better conditioned)::

              (P_NM.T@P_NM + 1)^(-1) P_NM.T@Y

        * QR: TBD
        * solve: Uses `scipy.linalg.solve` for the normal equations::

              (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y

        * lstsq: require rcond value. Use `numpy.linalg.solve(rcond=rcond)` for the normal equations::

             (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y
    """

    def __init__(
        self, KMM, regularizer=1, jitter=0, solver="RKHS", relative_jitter=True
    ):
        self.solver = solver
        self.KMM = KMM
        self.relative_jitter = relative_jitter

        self._nM = len(KMM)
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            self._vk, self._Uk = scipy.linalg.eigh(KMM)
            self._vk = self._vk[::-1]
            self._Uk = self._Uk[:, ::-1]
        elif self.solver == "QR" or self.solver == "solve" or self.solver == "lstsq":
            # gets maximum eigenvalue of KMM to scale the numerical jitter
            self._KMM_maxeva = scipy.sparse.linalg.eigsh(
                KMM, k=1, return_eigenvectors=False
            )[0]
        else:
            raise ValueError(
                f"Solver {solver} not supported. Possible values "
                "are 'RKHS', 'RKHS-QR', 'QR', 'solve' or lstsq."
            )
        if relative_jitter:
            if self.solver == "RKHS" or self.solver == "RKHS-QR":
                self._jitter_scale = self._vk[0]
            elif (
                self.solver == "QR" or self.solver == "solve" or self.solver == "lstsq"
            ):
                self._jitter_scale = self._KMM_maxeva
        else:
            self._jitter_scale = 1.0
        self.set_regularizers(regularizer, jitter)

    def set_regularizers(self, regularizer=1.0, jitter=0.0):
        self.regularizer = regularizer
        self.jitter = jitter
        if self.solver == "RKHS" or self.solver == "RKHS-QR":
            self._nM = len(np.where(self._vk > self.jitter * self._jitter_scale)[0])
            self._PKPhi = self._Uk[:, : self._nM] * 1 / np.sqrt(self._vk[: self._nM])
        elif self.solver == "QR":
            self._VMM = scipy.linalg.cholesky(
                self.regularizer * self.KMM
                + np.eye(self._nM) * self._jitter_scale * self.jitter
            )
        self._Cov = np.zeros((self._nM, self._nM))
        self._KY = None

    def fit(self, KNM, Y, rcond=None):
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]
        if self.solver == "RKHS":
            Phi = KNM @ self._PKPhi
            self._weights = self._PKPhi @ scipy.linalg.solve(
                Phi.T @ Phi + np.eye(self._nM) * self.regularizer,
                Phi.T @ Y,
                assume_a="pos",
            )
        elif self.solver == "RKHS-QR":
            A = np.vstack(
                [KNM @ self._PKPhi, np.sqrt(self.regularizer) * np.eye(self._nM)]
            )
            Q, R = np.linalg.qr(A)
            self._weights = self._PKPhi @ scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((self._nM, Y.shape[1]))])
            )
        elif self.solver == "QR":
            A = np.vstack([KNM, self._VMM])
            Q, R = np.linalg.qr(A)
            self._weights = scipy.linalg.solve_triangular(
                R, Q.T @ np.vstack([Y, np.zeros((KNM.shape[1], Y.shape[1]))])
            )
        elif self.solver == "solve":
            self._weights = scipy.linalg.solve(
                KNM.T @ KNM
                + self.regularizer * self.KMM
                + np.eye(self._nM) * self.jitter * self._jitter_scale,
                KNM.T @ Y,
                assume_a="pos",
            )
        elif self.solver == "lstsq":
            self._weights = np.linalg.lstsq(
                KNM.T @ KNM
                + self.regularizer * self.KMM
                + np.eye(self._nM) * self.jitter * self._jitter_scale,
                KNM.T @ Y,
                rcond=rcond,
            )[0]
        else:
            ValueError("solver not implemented")

    def predict(self, KTM):
        return KTM @ self._weights
