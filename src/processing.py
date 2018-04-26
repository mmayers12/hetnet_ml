import numpy as np
from scipy.sparse import issparse


class NotFittedError(Exception):
    """An error for a transform that has not been fitted"""
    def __init__(self, *args):
        self.message = "This {} instance is not fitted yet. " + \
                       "Call 'fit' with the appropriate arguments before using this method."


class Transform(object):
    """Base class for transformations containing no actual transformation of the data"""

    def __init__(self, standardize=True):
        self.transformed_mean = None
        self.transformed_std = None
        self.fit_ = False
        self.standardize = standardize

    def _fit_untransformed_data(self, data):
        """Empty Helper Function, Fits that need to be made prior to transformation"""
        pass

    def _transformation_function(self, data):
        """Empty helper function, how the data will be transformed by various subclasses"""
        return data

    def _fit_transformed_data(self, data):
        """Fits that need to be made after transformation, in this case the mean and std are stored"""

        if issparse(data):
            # Allow for use of sparse matrices
            self.transformed_mean = data.mean(axis=0).A[0]

            # Standard dev is tricky to get fast/efficiently
            def get_col_std(col):
                N = col.shape[0]
                sqr = col.copy()  # take a copy of the col
                sqr.data **= 2  # square the data, i.e. just the non-zero data
                variance = sqr.sum() / N - col.mean() ** 2
                return np.sqrt(variance)

            stds = []
            for i in range(data.shape[1]):
                stds.append(get_col_std(data.getcol(i)))

            self.transformed_std = np.array(stds)

        else:
            self.transformed_mean = data.mean(axis=0)
            self.transformed_std = data.std(axis=0, ddof=1)
        # Extract ndarray if original values were passed in a DataFrame
        try:
            self.transformed_std = self.transformed_std.values
        except:
            pass
        # Set 0 transformed_std to 1 so no NaN values are returned
        self.transformed_std[np.where(self.transformed_std == 0)] = 1

    def _standardize(self, data):
        """standardize the transformed data"""

        if issparse(data):
            data = data.tolil()
            data -= self.transformed_mean
            return data.multiply(self.transformed_std**-1).tocsc()

        return (data - self.transformed_mean) / self.transformed_std

    def __fit(self, data):
        self._fit_untransformed_data(data)
        trans_data = self._transformation_function(data)
        self._fit_transformed_data(trans_data)
        self.fit_ = True
        return trans_data

    def fit(self, data):
        """Fit the transformation to the current data"""
        self.__fit(data)

    def fit_transform(self, data):
        """Fit the transformation to the current data and return transformed data"""
        trans_data = self.__fit(data)
        if self.standardize:
            return self._standardize(trans_data)
        return trans_data

    def transform(self, data):
        """Transform data based on previous fit"""
        if not self.fit_:
            raise NotFittedError(NotFittedError().message.format(type(self).__name__))

        if self.standardize:
            return self._standardize(self._transformation_function(data))

        return self._transformation_function(data)


class DegreeTransform(Transform):
    """
    Transforms degree features

    For each feature the transformation is arcsinh(value / feaure_mean), which is then standardized to the
    mean and standard deviation of the transformed values.
    """

    def _transformation_function(self, data):
        return np.arcsinh(data)


class DWPCTransform(Transform):
    """
    Transforms DWPC features

    For each feature the transformation is arcsinh(value / feaure_mean), which is then standardized to the
    mean and standard deviation of the transformed values.
    """

    def __init__(self, standardize=False):
        self.initial_mean = None
        Transform.__init__(self, standardize)

    def _fit_untransformed_data(self, data):
        if issparse(data):
            self.initial_mean = data.mean(axis=0).A[0]
        else:
            self.initial_mean = data.mean(axis=0)

        # If input was DataFrame, Converts resultant series to ndarray
        try:
            self.initial_mean = self.initial_mean.values
        except:
            pass
        self.initial_mean[np.where(self.initial_mean == 0)] = 1

    def _transformation_function(self, data):
        if issparse(data):
            return np.arcsinh(data.multiply(self.initial_mean**-1))
        return np.arcsinh(data / self.initial_mean)
