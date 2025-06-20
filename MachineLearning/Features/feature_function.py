"""This module provides the class FeatureFunction"""


class FeatureFunction:
    """This class is to provide a modular possibility to execute any feature function"""
    def __init__(self, func, default_params=None):
        """
        Wraps a feature function with optional default parameters.

        :param func: The actual feature function (signal -> value)
        :param default_params: A dictionary of default parameters for the function.
        """
        self.func = func
        self.default_params = default_params or {}

    def __call__(self, obj, **override_params):
        """
        Executes the feature function with merged parameters.

        :param obj: The object, that provides the feature function.
        :param override_params: Optional parameters to override the defaults.
        :returns: Feature value.
        """
        params = {**self.default_params, **override_params}
        return self.func(obj, **params)
