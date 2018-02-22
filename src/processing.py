import numpy as np


class NotFittedError(Exception):
    """An error for a transform that has not been fitted"""
    def __init__(self, *args):
        self.message = "This {} instance is not fitted yet. " + \
                       "Call 'fit' with the appropriate arguments before using this method."


class Transform(object):
    """Base class for transformations containing no actual transformation of the data"""

    def __init__(self):
        self.transformed_mean = None
        self.transformed_std = None
        self.fit_ = False

    def _fit_untransformed_data(self, data):
        """Empty Helper Function, Fits that need to be made prior to transformation"""
        pass

    def _transformation_function(self, df):
        """Empty helper function, how the data will be transformed by various subclasses"""
        return df

    def _fit_transformed_data(self, data):
        """Fits that need to be made after transformation, in this case the mean and std are stored"""
        self.transformed_mean = data.mean(axis=0)
        self.transformed_std = data.std(axis=0, ddof=1)
        # Extract ndarray if original values were passed in a DataFrame
        try:
            self.transformed_std = self.transformed_std.values
        except:
            pass
        # Set 0 transformed_std to 1 so no NaN values are returned
        self.transformed_std[np.where(self.transformed_std == 0)] = 1

    def _standardize(self, df):
        """standardize the transformed data"""
        return (df - self.transformed_mean) / self.transformed_std

    def __fit(self, df):
        self._fit_untransformed_data(df)
        trans_data = self._transformation_function(df)
        self._fit_transformed_data(trans_data)
        self.fit_ = True
        return trans_data

    def fit(self, df):
        """Fit the transformation to the current data"""
        self.__fit(df)

    def fit_transform(self, df):
        """Fit the transformation to the current data and return transformed data"""
        trans_data = self.__fit(df)
        return self._standardize(trans_data)

    def transform(self, df):
        """Transform data based on previous fit"""
        if not self.fit_:
            raise NotFittedError(NotFittedError().message.format(type(self).__name__))

        return self._standardize(self._transformation_function(df))


class DegreeTransform(Transform):
    """
    Transforms degree features

    For each feature the transformation is arcsinh(value / feaure_mean), which is then standardized to the
    mean and standard deviation of the transformed values.
    """

    def _transformation_function(self, df):
        return np.arcsinh(df)


class DWPCTransform(Transform):
    """
    Transforms DWPC features

    For each feature the transformation is arcsinh(value / feaure_mean), which is then standardized to the
    mean and standard deviation of the transformed values.
    """

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

    def _transformation_function(self, df):
        return np.arcsinh(df / self.initial_mean)
