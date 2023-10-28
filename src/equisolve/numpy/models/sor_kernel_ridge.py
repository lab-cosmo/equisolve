from typing import List, Union

import metatensor
import numpy as np
import scipy
from metatensor import Labels, TensorBlock, TensorMap

from ..kernels import AggregateKernel, AggregateLinear, AggregatePolynomial


class SorKernelRidge:
    """
    Uses the subset of regressors (SoR) approximation of the kernel

    .. math::

        w = (k_{nm}K_{nm} + k_{mm})^-1 @ k_{mn} y

    plus regularization

    Reference
    ---------
    Quinonero-Candela, J., & Rasmussen, C. E. (2005). A unifying view of sparse
    approximate Gaussian process regression. The Journal of Machine Learning Research,
    6, 1939-1959.
    """

    def __init__(
        self,
    ) -> None:
        self._weights = None

    def _set_kernel(self, kernel: Union[str, AggregateKernel], **kernel_kwargs):
        valid_kernels = ["linear", "polynomial", "precomputed"]
        aggregate_type = kernel_kwargs.get("aggregate_type", "sum")
        if aggregate_type != "sum":
            raise ValueError(
                f'aggregate_type={aggregate_type!r} found but must be "sum"'
            )
        if kernel == "linear":
            self._kernel = AggregateLinear(aggregate_type="sum", **kernel_kwargs)
        elif kernel == "polynomial":
            self._kernel = AggregatePolynomial(aggregate_type="sum", **kernel_kwargs)
        elif kernel == "precomputed":
            self._kernel = None
        elif isinstance(kernel, AggregateKernel):
            self._kernel = kernel
        else:
            raise ValueError(
                f"kernel type {kernel!r} is not supported. Please use one "
                f"of the valid kernels {valid_kernels!r}"
            )

    def fit(
        self,
        X: TensorMap,
        X_pseudo: TensorMap,
        y: TensorMap,
        kernel_type: Union[str, AggregateKernel] = "linear",
        kernel_kwargs: dict = None,
        accumulate_key_names: Union[str, List[str], None] = None,
        alpha: Union[float, TensorMap] = 1.0,
        solver: str = "RKHS-QR",
        rcond: float = None,
    ):
        r"""
        :param X:
            features
            if kernel type "precomputed" is used, the kernel k_nm is assumed
        :param X_pseudo:
            pseudo points
            if kernel type "precomputed" is used, the kernel k_mm is assumed
        :param y:
            targets
        :param kernel_type:
            type of kernel used
        :param kernel_kwargs:
            additional keyword argument for specific kernel
            - **linear** None
            - **polynomial** degree
        :param accumulate_key_names:
            a string or list of strings that specify which key names should be
            accumulate to one kernel. This is intended for key columns inducing sparsity
            in the properties (e.g. neighbour species)
        :param alpha:
            regularization
        :param solver:
            determines which solver to use
            ... TODO doc ...
        :param rcond:
            argument for the solver lstsq


        TODO move to developer doc

        Derivation
        ----------

        We take equation (16b) (the mean expression)

        .. math::

            \sigma^{-2} K_{tm}\Sigma K_{MN}y

        we put in the $\Sigma$

        .. math::

            \sigma^{-2} K_{tm}(\sigma^{-2}K_{mn}K_{mn}+K_{mm})^{-1} K_{mn}y

        We can move around the $\sigma's$

        .. math::

             K_{tm}((K_{mn}\sigma^{-1})(\sigma^{-1}K_{mn)}+K_{mm})^{-1}
                            (K_{mn}\sigma^{-1})(y\sigma^{-1})

        you can see the building blocks in the code are $K_{mn}\sigma^{-1}$ and
        $y\sigma^{-1}$
        """

        # TODO store backend to use bring it back to original backend (e.g. torch)
        #      at the end of calculation
        X = metatensor.to(X, backend="numpy")
        X_pseudo = metatensor.to(X_pseudo, backend="numpy")
        y = metatensor.to(y, backend="numpy")

        if type(alpha) is float:
            alpha_is_zero = alpha == 0.0

            alpha_tensor = metatensor.ones_like(y)

            samples = Labels(
                names=y.samples_names,
                values=np.zeros([1, len(y.samples_names)], dtype=int),
            )

            alpha_tensor = metatensor.slice(
                alpha_tensor, axis="samples", labels=samples
            )
            alpha = metatensor.multiply(alpha_tensor, alpha)

        elif isinstance(alpha, TensorMap):
            raise NotImplementedError("TensorMaps are not yet supported")
        else:
            raise ValueError("alpha must either be a float or a TensorMap")

        if kernel_kwargs is None:
            kernel_kwargs = {}
        self._set_kernel(kernel_type, **kernel_kwargs)
        self._kernel_type = kernel_type

        if self._kernel_type == "precomputed":
            k_nm = X
            k_mm = X_pseudo
        else:
            k_mm = self._kernel(X_pseudo, X_pseudo, are_pseudo_points=(True, True))
            k_nm = self._kernel(X, X_pseudo, are_pseudo_points=(False, True))

        if accumulate_key_names is not None:
            raise NotImplementedError(
                "accumulate_key_names only supports None for the moment"
            )

        # solve
        weight_blocks = []
        for key, y_block in y.items():
            k_nm_block = k_nm.block(key)
            k_mm_block = k_mm.block(key)
            X_block = X.block(key)
            structures = metatensor.operations._dispatch.unique(
                k_nm_block.samples["structure"]
            )
            n_atoms_per_structure = []
            for structure in structures:
                # TODO dispatch sum
                n_atoms = np.sum(X_block.samples["structure"] == structure)
                n_atoms_per_structure.append(float(n_atoms))

            # PR COMMENT removed delta because would say this is part of the standardizr
            n_atoms_per_structure = np.array(n_atoms_per_structure)
            normalization = metatensor.operations._dispatch.sqrt(n_atoms_per_structure)
            alpha_values = alpha.block(key).values
            if not alpha_is_zero:
                normalization /= alpha_values[0, 0]
            normalization = normalization[:, None]

            k_nm_reg = k_nm_block.values * normalization
            y_reg = y_block.values * normalization

            self._solver = _SorKernelSolver(
                k_mm_block.values, regularizer=1, jitter=0, solver=solver
            )

            if rcond is None:
                # PR COMMENT maybe we get this from a case class RidgeBase Ridge
                #            uses the same
                rcond_ = max(k_nm_reg.shape) * np.finfo(k_nm_reg.dtype.char.lower()).eps
            else:
                rcond_ = rcond
            self._solver.fit(k_nm_reg, y_reg, rcond=rcond_)

            weight_block = TensorBlock(
                values=self._solver.weights.T,
                samples=y_block.properties,
                components=k_nm_block.components,
                properties=k_nm_block.properties,
            )
            weight_blocks.append(weight_block)

        self._weights = TensorMap(y.keys, weight_blocks)

        self._X_pseudo = X_pseudo.copy()

    @property
    def weights(self) -> TensorMap:
        return self._weights

    def predict(self, T: TensorMap) -> TensorMap:
        """
        :param T:
            features
            if kernel type "precomputed" is used, the kernel k_tm is assumed
        """
        if self._kernel_type == "precomputed":
            k_tm = T
        else:
            k_tm = self._kernel(T, self._X_pseudo, are_pseudo_points=(False, True))
        return metatensor.dot(k_tm, self._weights)

    def forward(self, tensor: TensorMap) -> TensorMap:
        return self.predict(tensor)


class _SorKernelSolver:
    """
    A few quick implementation notes, docs to be done.

    This is meant to solve the subset of regressors (SoR) problem::

    .. math::

        w = (KNM.T@KNM + reg*KMM)^-1 @ KNM.T@y

    The inverse needs to be stabilized with application of a numerical jitter,
    that is expressed as a fraction of the largest eigenvalue of KMM

    :param KMM:
        KNM matrix
    :param regularizer:
        regularizer
    :param jitter:
        numerical jitter to stabilize fit
    :param solver:
        Method to solve the sparse KRR equations.

        * RKHS-QR: TBD
        * RKHS: Compute first the reproducing kernel features by diagonalizing K_MM and
          computing `P_NM = K_NM @ U_MM @ Lam_MM^(-1.2)` and then solves the linear
          problem for those (which is usually better conditioned)::

              (P_NM.T@P_NM + 1)^(-1) P_NM.T@Y

        * QR: TBD
        * solve: Uses `scipy.linalg.solve` for the normal equations::

              (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y

        * lstsq: require rcond value. Use `numpy.linalg.solve(rcond=rcond)` for the
          normal equations::

             (K_NM.T@K_NM + K_MM)^(-1) K_NM.T@Y

    Reference
    ---------
    Foster, L., Waagen, A., Aijaz, N., Hurley, M., Luis, A., Rinsky, J., ... &
    Srivastava, A. (2009). Stable and Efficient Gaussian Process Calculations. Journal
    of Machine Learning Research, 10(4).
    """

    def __init__(
        self,
        KMM: np.ndarray,
        regularizer: float = 1.0,
        jitter: float = 0.0,
        solver: str = "RKHS",
        relative_jitter: bool = True,
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

    def partial_fit(self, KNM, Y, accumulate_only=False, rcond=None):
        if len(Y) > 0:
            # only accumulate if we are passing data
            if len(Y.shape) == 1:
                Y = Y[:, np.newaxis]
            if self.solver == "RKHS":
                Phi = KNM @ self._PKPhi
            elif self.solver == "solve" or self.solver == "lstsq":
                Phi = KNM
            else:
                raise ValueError(
                    "Partial fit can only be realized with "
                    "solver = 'RKHS' or 'solve'"
                )
            if self._KY is None:
                self._KY = np.zeros((self._nM, Y.shape[1]))

            self._Cov += Phi.T @ Phi
            self._KY += Phi.T @ Y

        # do actual fit if called with empty array or if asked
        if len(Y) == 0 or (not accumulate_only):
            if self.solver == "RKHS":
                self._weights = self._PKPhi @ scipy.linalg.solve(
                    self._Cov + np.eye(self._nM) * self.regularizer,
                    self._KY,
                    assume_a="pos",
                )
            elif self.solver == "solve":
                self._weights = scipy.linalg.solve(
                    self._Cov
                    + self.regularizer * self.KMM
                    + np.eye(self.KMM.shape[0]) * self.jitter * self._jitter_scale,
                    self._KY,
                    assume_a="pos",
                )
            elif self.solver == "lstsq":
                self._weights = np.linalg.lstsq(
                    self._Cov
                    + self.regularizer * self.KMM
                    + np.eye(self.KMM.shape[0]) * self.jitter * self._jitter_scale,
                    self._KY,
                    rcond=rcond,
                )[0]

    @property
    def weights(self):
        return self._weights

    def predict(self, KTM):
        return KTM @ self._weights
