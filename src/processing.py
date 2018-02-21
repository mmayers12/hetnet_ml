import numpy as np


class Transform(object):
    """Base class for transformations containing no actual transformation of the data"""

    def __init__(self):
        self.transformed_mean = None
        self.transformed_std = None

    def fit(self, df):
        """Fit the transformation to the current data"""
        trans_data = self._transformation_function(df)
        self._fit_transformed_data(trans_data)

    def _fit_transformed_data(self, data):
        self.transformed_mean = data.mean(axis=0)
        self.transformed_std = data.std(axis=0, ddof=1)
        try:
            self.transformed_std = self.transformed_std.values
        except:
            pass
        # Set 0 transformed_std to 1 so no NaN values are returned
        self.transformed_std[np.where(self.transformed_std == 0)] = 1

    def fit_transform(self, df):
        """Fit the transformation to the current data and return transformed data"""
        trans_data = self._transformation_function(df)
        self._fit_transformed_data(trans_data)
        return self._standardize(trans_data)

    def transform(self, df):
        """Transform data based on previous fit"""
        assert self.transformed_mean is not None and self.transformed_std is not None
        return self._standardize(self._transformation_function(df))

    def _standardize(self, df):
        return (df - self.transformed_mean) / self.transformed_std

    def _transformation_function(self, df):
        """Empty helper function"""
        return df


class DegreeTransform(Transform):
    """Transforms degree features"""

    def _transformation_function(self, df):
        return np.arcsinh(df)


class DWPCTransform(Transform):
    "Transforms DWPC features"

    def __init__(self):
        self.initial_mean = None
        Transform.__init__(self)

    def _fit_untransformed_data(self, data):
        self.initial_mean = data.mean(axis=0)
        # If input was DataFrame, Converts resultant series to ndarray
        try:
            self.initial_mean = self.initial_mean.values
        except:
            pass
        self.initial_mean[np.where(self.initial_mean == 0)] = 1

    def fit_transform(self, df):
        """Fit the transformation to the current data and return transformed data"""
        trans_data = self._transformation_function(df)
        self._fit_transformed_data(trans_data)
        return self._standardize(trans_data)

    def transform(self, df):
        assert self.initial_mean is not None
        return Transform.transform(self, df)

    def _transformation_function(self, df):
        if self.initial_mean is None:
            self._fit_untransformed_data(df)
        return np.arcsinh(df / self.initial_mean)