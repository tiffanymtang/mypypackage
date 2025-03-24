import unittest
import pandas as pd
import seaborn as sns
from mypypackage.plot import plot_pairs

class TestPlotPairs(unittest.TestCase):

    def setUp(self):
        self.data = sns.load_dataset('iris')
        self.vars = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    def test_plot_pairs_no_color(self):
        """Test plot_pairs without color_by parameter"""
        g = plot_pairs(self.data, self.vars)
        self.assertIsInstance(g, sns.axisgrid.PairGrid)

    def test_plot_pairs_with_color(self):
        """Test plot_pairs with color_by parameter"""
        g = plot_pairs(self.data, self.vars, color_by='species')
        self.assertIsInstance(g, sns.axisgrid.PairGrid)

    def test_plot_pairs_invalid_color(self):
        """Test plot_pairs with an invalid color_by parameter"""
        with self.assertRaises(KeyError):
            plot_pairs(self.data, self.vars, color_by='invalid_column')

    def test_plot_pairs_corr_fontsize(self):
        """Test plot_pairs with different corr_fontsize"""
        g = plot_pairs(self.data, self.vars, corr_fontsize=15)
        self.assertIsInstance(g, sns.axisgrid.PairGrid)

    def test_plot_pairs_additional_kwargs(self):
        """Test plot_pairs with additional keyword arguments"""
        g = plot_pairs(self.data, self.vars, diag_kind='kde')
        self.assertIsInstance(g, sns.axisgrid.PairGrid)

if __name__ == '__main__':
    unittest.main()