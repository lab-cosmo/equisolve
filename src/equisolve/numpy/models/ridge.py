from equistore import Labels, TensorBlock, TensorMap
from sklearn.linear_model import Ridge, RidgeCV


class EquiRidge:
    """ """

    def __init__(self, regressor_dictionary):
        self.regressor_dictionary = {}
        self.fitted_properties_ = {}

        for key, regressor in regressor_dictionary.items():
            if isinstance(regressor, Ridge):
                self.regressor_dictionary[key] = regressor
            elif isinstance(regressor, dict):
                self.regressor_dictionary[key] = Ridge(**regressor)
            else:
                raise ValueError

    def fit(self, X, y):
        for key in X.keys:
            if key not in y.keys:
                raise ValueError("X and y must have the same keys.")

        for key in X.keys:
            if key not in self.regressor_dictionary:
                raise ValueError("You must supply a regression type for every key.")

        for key in X.keys:
            self.regressor_dictionary[key].fit(X.block(key).values, y.block(key).values)
            self.fitted_properties_[key] = y.block(key).properties
        return self

    def predict(self, X):
        blocks = []
        for key in X.keys:
            blocks.append(
                TensorBlock(
                    values=self.regressor_dictionary[key].predict(X.block(key).values),
                    samples=X.block(key).samples,
                    components=X.block(key).components,
                    properties=self.fitted_properties_[key],
                )
            )
        return TensorMap(X.keys, blocks)
