# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:04:06 2024

@author: Simon
"""

# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
import scipy
from scipy import io
from tqdm import tqdm
from tdlm import plotting
import matplotlib.pyplot as plt


class TestPlotting(unittest.TestCase):

    def test_uperms(self):

        # dummy curve, should be below threshold
        seq_fwd = (np.arange(30*3) + 5).reshape([1, 3, 30]).astype(float)
        seq_fwd[:, :, 0] = np.nan
        plotting.plot_sequenceness(seq_fwd, seq_fwd, which = ['fwd'])

        # dummy curve, should be above threshold
        seq_fwd = (np.arange(30*3) + 5)[::-1].reshape([1, 3, 30]).astype(float)
        seq_fwd[:, :, 0] = np.nan
        plotting.plot_sequenceness(seq_fwd, seq_fwd, which = ['fwd'])

    def test_plot_sequenceness_signflip_two_subjects(self):
        """Test signflip plotting path with sufficient observations"""
        rng = np.random.default_rng(42)
        seq_fwd = rng.standard_normal((2, 4, 30))
        seq_bkw = rng.standard_normal((2, 4, 30))
        seq_fwd[:, :, 0] = np.nan
        seq_bkw[:, :, 0] = np.nan

        fig, ax = plt.subplots()
        out_ax = plotting.plot_sequenceness(
            seq_fwd,
            seq_bkw,
            which=['fwd'],
            plotsignflip=True,
            ax=ax,
        )
        self.assertIs(out_ax, ax)
        plt.close(fig)

    def test_plot_sequenceness_signflip_single_subject(self):
        """Single-subject plotting should not crash when signflip is requested"""
        rng = np.random.default_rng(0)
        seq_fwd = rng.standard_normal((1, 4, 30))
        seq_bkw = rng.standard_normal((1, 4, 30))
        seq_fwd[:, :, 0] = np.nan
        seq_bkw[:, :, 0] = np.nan

        fig, ax = plt.subplots()
        out_ax = plotting.plot_sequenceness(
            seq_fwd,
            seq_bkw,
            which=['fwd'],
            plotsignflip=True,
            ax=ax,
        )
        self.assertIs(out_ax, ax)
        plt.close(fig)

if __name__=='__main__':
    unittest.main()
