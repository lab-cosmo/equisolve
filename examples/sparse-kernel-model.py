# Example

from equisolve.utils import ase_to_tensormap
from equisolve.numpy.models.sparse_kernel import compute_sparse_kernel, SparseKernelRidge

import equistore 

from equisolve.numpy.sample_section import FPS

y = ase_to_tensormap(frames)
ps = TensorMap()
n_to_select = 100
degree = 3


pseudo_points = FPS(n_to_select=n_to_select).fit_transform(ps)

k_mm = compute_sparse_kernel(tensor=pseudo_points, pseudo_points=pseudo_points, degree=degree)
k_nm = compute_sparse_kernel(tensor=ps, pseudo_points=pseudo_points, degree=degree)
k_y = compute_sparse_kernel(tensor=y, pseudo_points=pseudo_points, degree=degree)

# User can do their structure sum
k_nm = equistore.sum_over_samples(k_nm, samples_names="species_center")

clf = SparseKernelRidge()
clf.fit(k_mm=k_mm, k_nm=k_nm, y=k_y)