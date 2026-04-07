import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from abc import ABC, abstractmethod
from ipywidgets import interact, IntSlider
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D

class Distribution(ABC):
    def __init__(self, name="Generic Distribution"):
        self.name = name
        self.mc_data = np.array([])

        # Track limits for graph drawing
        self.x_min = 0
        self.x_max = 0
        self.y_max = 0.0

    def save_results(self, results):
        """Appends Monte Carlo results and dynamically updates axis limits."""
        self.mc_data = np.append(self.mc_data, results)

        # Refresh x_max and x_min based on the new dataset
        if len(self.mc_data) > 0:
            self.x_min = int(np.min(self.mc_data))
            self.x_max = int(np.max(self.mc_data))

            # To estimate y_max (highest probability density) we can look at the mode
            # This is a rough estimation for dynamic scaling
            values, counts = np.unique(self.mc_data, return_counts=True)
            self.y_max = np.max(counts) / len(self.mc_data)

    def clear_results(self):
        """Clears stored simulation data and resets limits."""
        self.mc_data = np.array([])
        self.x_min = 0
        self.x_max = 0
        self.y_max = 0.0

    @abstractmethod
    def simulate(self, n_trials):
        pass

    def support_start(self):
      """
      Left boundary of the support for summation in CDF.
      Override this in subclasses when needed.
      """
      return 0

    def has_pmf(self):
      return type(self).pmf is not Distribution.pmf

    def has_cdf(self):
      return type(self).cdf is not Distribution.cdf

    @abstractmethod
    def pmf(self, k):
      """
      Default PMF:
      if subclass did not implement PMF directly, try deriving it from CDF.
      """

      return self.FromCdf_ToPmf(k)


    @abstractmethod
    def cdf(self, k):
      """
      Default CDF:
      if subclass did not implement CDF directly, try deriving it from PMF.
      """

      return self.FromPmf_ToCdf(k)


    def FromCdf_ToPmf(self, k):
      if not self.has_cdf():
        raise NotImplementedError(f"{self.name}: cannot derive PMF because CDF is not implemented.")

      if int(k) != k:
        return 0.0

      k = int(k)
      return self.cdf(k) - self.cdf(k - 1)

    def FromPmf_ToCdf(self, k):
      if not self.has_pmf():
        raise NotImplementedError(f"{self.name}: cannot derive CDF because PMF is not implemented.")

      if k < self.support_start():
        return 0.0

      k = int(np.floor(k))
      start = self.support_start()
      return sum(self.pmf(x) for x in range(start, k + 1))


    def plot_pmf(self, show_theoretical=True):
      has_data = len(self.mc_data) > 0

      # FIX: Only require PMF/CDF if we are actually drawing the theoretical line
      if show_theoretical and not self.has_pmf() and not self.has_cdf():
        raise NotImplementedError(f"{self.name}: Neither PMF nor CDF is implemented.")

      # Safety check: If we have no math AND no data, we can't draw anything!
      if not show_theoretical and not has_data:
        raise ValueError(f"{self.name}: Run simulate() first, or set show_theoretical=True")

      """Draws the Probability Mass Function (Histogram vs Curve)."""
      fig, ax = plt.subplots(figsize=(8, 5))

      has_data = len(self.mc_data) > 0
      plot_x_min = self.x_min if has_data else self.support_start()
      plot_x_max = self.x_max if has_data else self.support_start() + 20

      if has_data:
          bins = np.arange(plot_x_min, plot_x_max + 2) - 0.5
          ax.hist(self.mc_data, bins=bins, density=True, alpha=0.6,
                  color='steelblue', edgecolor='black', label='Monte Carlo Data')

      if show_theoretical:
          x_vals = np.arange(plot_x_min, plot_x_max + 1)
          y_vals = [self.pmf(x) for x in x_vals]
          ax.plot(x_vals, y_vals, 'ro-', markersize=5, label='Theoretical PMF', linewidth=2)

          if self.y_max == 0 and len(y_vals) > 0 and max(y_vals) > 0:
            ax.set_ylim(0, max(y_vals) * 1.2)

      # Utilize the saved y_max to add some padding to the top of the graph
      if self.y_max > 0:
          ax.set_ylim(0, self.y_max * 1.2)

      ax.set_title(f"{self.name} - PMF")
      ax.set_xlabel("Value")
      ax.set_ylabel("Probability")
      ax.legend()
      ax.grid(axis='y', alpha=0.4)
      plt.tight_layout()
      plt.show()


    def plot_cdf(self, show_theoretical=True):
      has_data = len(self.mc_data) > 0

      # FIX: Only require CDF/PMF if we are actually drawing the theoretical line
      if show_theoretical and not self.has_cdf() and not self.has_pmf():
        raise NotImplementedError(f"{self.name}: neither CDF nor PMF is implemented.")

      # Safety check
      if not show_theoretical and not has_data:
        raise ValueError(f"{self.name}: Run simulate() first, or set show_theoretical=True")

      fig, ax = plt.subplots(figsize=(8, 5))

      has_data = len(self.mc_data) > 0
      plot_x_min = self.x_min if has_data else self.support_start()
      plot_x_max = self.x_max if has_data else self.support_start() + 20

      if has_data:
          x_data = np.sort(self.mc_data)
          y_data = np.arange(1, len(x_data) + 1) / len(x_data)

          ax.step(x_data, y_data, where='post', label='Empirical CDF', color='steelblue', linewidth=2)

      if show_theoretical:
          x_vals = np.arange(plot_x_min, plot_x_max + 1)
          y_vals = [self.cdf(x) for x in x_vals]

          ax.step(x_vals, y_vals, where='post', label='Theoretical CDF', color='red', linestyle='--', linewidth=2)

      ax.set_title(f"{self.name} - CDF")
      ax.set_xlabel("Value")
      ax.set_ylabel("Cumulative Probability")
      ax.set_ylim(0, 1.05)
      ax.legend()
      ax.grid(axis='both', alpha=0.4)
      plt.tight_layout()
      plt.show()


    # Calculuating some statistics (from ChatGPT)
    def empirical_stats(self):
      if len(self.mc_data) == 0:
        raise ValueError(f"{self.name}: No simulation data available.")

      mean = np.mean(self.mc_data)
      var = np.var(self.mc_data)
      std = np.std(self.mc_data)

      # Avoid division by zero
      if std == 0:
        skew = 0.0
      else:
        skew = np.mean(((self.mc_data - mean) / std) ** 3)

      return {
        "mean": mean,
        "variance": var,
        "skewness": skew
      }

    def theoretical_stats(self):
      raise NotImplementedError(f"{self.name}: theoretical stats not implemented.")

    def compare_stats(self):
      emp = self.empirical_stats()

      print(f"\n{self.name} Statistics Comparison")
      print("-" * 45)

      try:
        theo = self.theoretical_stats()

        print(f"{'Metric':<12} {'Empirical':<15} {'Theoretical':<15}")
        print("-" * 45)

        for key in emp:
          print(f"{key:<12} {emp[key]:<15.5f} {theo[key]:<15.5f}")

      except NotImplementedError:
        print("Theoretical stats not available.\n")
        for key in emp:
          print(f"{key:<12}: {emp[key]:.5f}")

    def stats_table(self):
      emp = self.empirical_stats()

      try:
        theo = self.theoretical_stats()
        df = pd.DataFrame({
          "Empirical": emp,
          "Theoretical": theo
        })
      except NotImplementedError:
        df = pd.DataFrame({
          "Empirical": emp
        })

      return df

class DiscreteUniformDistribution(Distribution):
    def __init__(self, a, b):
        super().__init__(name=f"DiscreteUniform(a={a}, b={b})")
        self.a = a
        self.b = b

    def support_start(self):
        return self.a

    def simulate(self, n_trials):
        results = np.random.randint(self.a, self.b + 1, size=n_trials)
        self.save_results(results)
        return results

    def pmf(self, k):
        if int(k) != k:
            return 0.0

        k = int(k)
        if self.a <= k <= self.b:
            return 1 / (self.b - self.a + 1)
        return 0.0

    def cdf(self, k):
      return self.FromPmf_ToCdf(k)


    # Calculating stats
    def theoretical_stats(self):
      mean = (self.a + self.b) / 2
      var = ((self.b - self.a + 1)**2 - 1) / 12
      skew = 0.0

      return {
        "mean": mean,
        "variance": var,
        "skewness": skew
      }

class BinomialDistribution(Distribution):
    def __init__(self, n, p):
        super().__init__(name=f"Binomial(n={n}, p={p})")
        self.n = n
        self.p = p

    def simulate(self, n_trials):
      results = np.random.binomial(self.n, self.p, size=n_trials)
      self.save_results(results)
      return results

    def pmf(self, k):
      if k < 0 or k > self.n or int(k) != k:
        return 0.0

      k = int(k)
      return math.comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Theoretical CDF: Sum of PMFs up to k."""
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        # Sum the PMF for all integers from 0 up to floor(k)
        return sum(self.pmf(i) for i in range(int(k) + 1))


    # Calculating stats
    def theoretical_stats(self):
      mean = self.n * self.p
      var = self.n * self.p * (1 - self.p)

      if var == 0:
        skew = 0.0
      else:
        skew = (1 - 2 * self.p) / math.sqrt(var)

      return {
        "mean": mean,
        "variance": var,
        "skewness": skew
      }
      
#print(BinomialDistribution(20,0.1).theoretical_stats())
