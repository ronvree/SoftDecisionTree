import math


class EMA:

    """
    Simple data structure for storing an exponentially decaying moving average (EMA)
    """

    def __init__(self, alpha: float, initial_value: float = None):
        """
        Create a new EMA
        :param alpha: the decay coefficient
        :param initial_value: optional initial value for the EMA. First input is taken by default
        """
        self.alpha = alpha
        if initial_value is None:
            self._value = float('nan')
        else:
            self._value = initial_value

    def __repr__(self):
        return f'EMA: {self._value} (alpha: {self.alpha})'

    def add(self, x: float, update: bool = True):
        """
        Add a value to the moving average
        :param x: the value to be added
        :param update: indicates whether the internal EMA value should be updated (True by default)
        :return: the new EMA value
        """
        # Compute the new EMA value
        if self.is_empty():
            value = x
        else:
            value = self.alpha * x + (1 - self.alpha) * self._value
        if update:  # Store result if necessary
            self._value = value
        return value

    def add_all(self, xs: iter, update: bool = True):
        """
        Add multiple values to the moving average
        :param xs: a sequence of values to be added
        :param update: indicates whether the internal EMA value should be updated (True by default)
        :return: the new EMA value
        """
        for x in xs:
            self.add(x, update=update)
        return self.value

    @property
    def value(self) -> float:
        """
        Get the EMA's value
        :return: the current value of the EMA
        """
        if self.is_empty():
            raise Exception('No window to take EMA from!')
        return self._value

    def is_empty(self) -> bool:
        """
        Checks if the EMA has seen data points to make an estimate from
        :return: a boolean indicating whether the EMA's value is defined
        """
        return math.isnan(self._value)

