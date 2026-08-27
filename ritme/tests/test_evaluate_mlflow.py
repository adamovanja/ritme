import unittest

import pandas as pd

from ritme.evaluate_mlflow import create_color_map


class TestCreateColorMap(unittest.TestCase):
    def test_maps_unique_values_to_colors(self):
        # Pins the matplotlib >= 3.9 compatible colormap lookup
        # (matplotlib.cm.get_cmap was removed).
        df = pd.DataFrame({"model": ["xgb", "linreg", "xgb", "trac"]})
        colors, color_map = create_color_map(df, "model")
        self.assertEqual(len(colors), len(df))
        self.assertEqual(set(color_map), {"xgb", "linreg", "trac"})
        self.assertEqual(colors[0], colors[2])


if __name__ == "__main__":
    unittest.main()
