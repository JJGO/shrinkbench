import numpy as np


class OnlineStats:
    """
    Welford's algorithm to compute sample mean and sample variance incrementally.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """

    def __init__(self, iterable=None):
        """Online Mean and Variance from single samples

        Running stats,
        This is compatible with np.ndarray objects and as long as the

        Keyword Arguments:
            iterable {[iterable} -- Values to initialize (default: {None})
        """
        self.n = 0
        self.mean = 0.0
        self.S = 0.0
        if iterable is not None:
            self.addN(iterable)

    def add(self, datum):
        """Add a single datum

        Internals are updated using Welford's method

        Arguments:
            datum  -- Numerical object
        """
        self.n += 1
        delta = datum - self.mean
        # Mk = Mk-1+ (xk – Mk-1)/k
        self.mean += delta / self.n
        # Sk = Sk-1 + (xk – Mk-1)*(xk – Mk).
        self.S += delta * (datum - self.mean)

    def addN(self, iterable, batch=False):
        """Add N data to the stats

        Arguments:
            iterable {[type]} -- [description]

        Keyword Arguments:
            batch {bool} -- If true, then the mean and std are computed over
            the new array using numpy and then that updates the current stats
        """
        if batch:
            add = self + OnlineStats.from_values(len(iterable), np.mean(iterable), np.std(iterable), 0)
            self.n, self.mean, self.S = add.n, add.mean, add.S
        else:
            for datum in iterable:
                self.add(datum)

    def pop(self, datum):
        if self.n == 0:
            raise ValueError("Stats must be non empty")

        self.n -= 1
        delta = datum - self.mean
        # Mk-1 = Mk - (xk - Mk) / (k - 1)
        self.mean -= delta / self.n
        # Sk-1 = Sk - (xk – Mk-1) * (xk – Mk)
        self.S -= (datum - self.mean) * delta

    def popN(self, iterable, batch=False):
        raise NotImplementedError

    @property
    def variance(self):
        # For 2 ≤ k ≤ n, the kth estimate of the variance is s2 = Sk/(k – 1).
        return self.S / self.n

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def flatmean(self):
        # for datapoints which are arrays
        return np.mean(self.mean)

    @property
    def flatvariance(self):
        # for datapoints which are arrays
        return np.mean(self.variance+self.mean**2) - self.flatmean**2

    @property
    def flatstd(self):
        return np.sqrt(self.flatvariance)

    @staticmethod
    def from_values(n, mean, std):
        stats = OnlineStats()
        stats.n = n
        stats.mean = mean
        stats.S = std**2 * n
        return stats

    @staticmethod
    def from_raw_values(n, mean, S):
        stats = OnlineStats()
        stats.n = n
        stats.mean = mean
        stats.S = S
        return stats

    def __str__(self):
        return f"n={self.n}  mean={self.mean}  std={self.std}"

    def __repr__(self):
        return f"OnlineStats.from_values(" + \
               f"n={self.n}, mean={self.mean}, " + \
               f"std={self.std})"

    def __add__(self, other):
        """Adding can be done with int|float or other Online Stats

        For other int|float, it is added to all previous values

        Arguments:
            other {[type]} -- [description]

        Returns:
            OnlineStats -- New instance with the sum.

        Raises:
            TypeError -- If the type is different from int|float|OnlineStas
        """
        if isinstance(other, OnlineStats):
            # Add the means, variances and n_samples of two objects
            n1, n2 = self.n, other.n
            mu1, mu2 = self.mean, other.mean
            S1, S2 = self.S, other.S
            # New stats
            n = n1 + n2
            mu = n1/n * mu1 + n2/n * mu2
            S = (S1 + n1 * mu1*mu1) + (S2 + n2 * mu2*mu2) - n * mu*mu
            return OnlineStats.from_raw_values(n, mu, S)
        if isinstance(other, (int, float)):
            # Add a fixed amount to all values. Only changes the mean
            return OnlineStats.from_raw_values(self.n, self.mean+other, self.S)
        else:
            raise TypeError("Can only add other groups or numbers")

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, k):
        # Multiply all values seen by some constant
        return OnlineStats.from_raw_values(self.n, self.mean*k, self.S*k**2)


class OnlineStatsMap:

    def __init__(self, *keys):
        self.stats = {}
        if keys is not None:
            self.register(*keys)

    def register(self, *keys):
        for k in keys:
            if k not in self.stats:
                self.stats[k] = OnlineStats()

    def __str__(self):
        s = "Stats"
        for k in self.stats:
            s += f'  {k}:  {str(self.stats[k])}'
