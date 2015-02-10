__author__ = 'marrabld'

import os
import sys

sys.path.append("../..")

import logger as log
import scipy
import scipy.fftpack
import numpy as np
import numpy.random
import libbootstrap
import libbootstrap.state
import csv
import spectralmodel

DEBUG_LEVEL = libbootstrap.state.State().debug
lg = log.logger
lg.setLevel(DEBUG_LEVEL)


class GenerateRealisation():
    def __init__(self, spectral_model, num_spectra):
        """

        :param object:
        :param num_spectra:
        :return:
        """

        try:
            assert(isinstance(spectral_model, spectralmodel.BuildSpectralModel))
        except:
            lg.exception()

        self._sm = spectral_model

    def _gen_random_numbers(self, num_gens):
        """

        :param num_gens:
        :return:
        """
        rand_nums = []

        length = self._sm.Syy.shape[0]

        for i_iter in range(0, num_gens):
            rand_nums.append(np.random.uniform(0, 1, length))

        self.rand_nums = np.asarray(rand_nums, float)

        return self.rand_nums

    def _calc_fft_random_numbers(self):
        """

        :param rand_nums:
        :return:
        """
        N = []

        for row in range(0, self.rand_nums.shape[0]):
            N.append(scipy.fftpack.fft(self.rand_nums[row]))

        self.N = np.asarray(N)

        return self.N

    def _calc_rand_powerspectrum(self):
        """

        :return:
        """

        self.Y = scipy.zeros_like(self.N)
        self.Y = self.N * scipy.sqrt(self._sm.Syy)

        return self.Y

    def _calc_inv_rand_powerspectrum(self):
        """

        :return:
        """

        y = []

        for row in range(0, self.Y.shape[0]):
            y.append(scipy.fftpack.ifft(self.Y[row]))

        self.y = scipy.asarray(y)

        return self.y

    def _calc_normalised_difference_inv_rand_powerspectrum(self):
        """

        :return:
        """

        self.y_std = scipy.std(self.y, 0)
        self.y_hat = self.y / self.y_std

        return self.y_hat

    def _calc_z(self):
        """

        :return:
        """

        self.z = self.y_hat * self._sm.std_delta_Rrs

        return self.z

    def gen_Rrs(self, mean_Rrs=None):
        """

        :return:
        """
        if mean_Rrs == None:
            # make the realizations from the mean of the model
            self.Rrs = self.z + self._sm.mean_Rrs
        else:
            self.Rrs = self.z + mean_Rrs

        return self.Rrs
