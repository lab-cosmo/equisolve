from ridge import EquiRidge
import equistore.io
from utils import structure_sum
import unittest


class LinearModelTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = equistore.io.load("power-spectrum.npz")
        self.y = equistore.io.load("energies.npz")
        self.X.keys_to_samples("species_center")
        self.X = structure_sum(self.X)
        self.regressors = {k: {"alpha": 1e-2} for k in self.X.keys}

    def test_wrong_y_keys(self):

        X = equistore.io.load("power-spectrum.npz")
        X = structure_sum(X)

        eridge = EquiRidge(self.regressors)
        with self.assertRaises(ValueError) as cm:
            eridge.fit(X, self.y)
            self.assertEquals(cm.message, "X and y must have the same keys.")

    def test_wrong_regression_keys(self):

        eridge = EquiRidge({(1,): {"alpha": 1e-2}})
        with self.assertRaises(ValueError) as cm:
            eridge.fit(self.X, self.y)
            self.assertEquals(
                cm.message, "You must supply a regression type for every key."
            )

    def test_pass(self):
        eridge = EquiRidge(self.regressors)
        eridge.fit(self.X, self.y)
        eridge.predict(self.X)


if __name__ == "__main__":
    unittest.main()
