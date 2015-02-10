__author__ = 'marrabld'

__author__ = 'marrabld'

import os
import sys

sys.path.append("../..")

import logger as log
import scipy
import scipy.fftpack
import numpy as np
import libbootstrap
import libbootstrap.state
import csv

DEBUG_LEVEL = libbootstrap.state.State().debug
lg = log.logger
lg.setLevel(DEBUG_LEVEL)


class BuildSpectralModel():
    """

    """

    def __init__(self, datafile):
        """

        :return:
        """
        self.wave, self.Rrs = helper_methods.read_iop_from_file(datafile)

    def _detrend(self):
        """


        :return:
        """
        self.mean_Rrs = scipy.mean(self.Rrs, 0)
        self.delta_Rrs = self.Rrs - self.mean_Rrs
        return self.delta_Rrs

    def _calc_std_delta_Rrs(self):
        """

        :return:
        """
        self.std_delta_Rrs = scipy.std(self.delta_Rrs, 0)
        return self.std_delta_Rrs

    def _calc_normalised_difference_Rrs(self):
        """

        :return:
        """
        self.delta_hat_Rrs = self.delta_Rrs / self.std_delta_Rrs
        return self.delta_hat_Rrs

    def _calc_power_spectrum(self):
        """

        :return:
        """
        Y = []
        for row in self.delta_hat_Rrs:
            Y.append(scipy.fftpack.fft(row))

        Y = scipy.asarray(Y)
        self.Syy = scipy.mean(Y * np.conj(Y), 0)
        return self.Syy

    def build(self):
        """

        :return:
        """
        self._detrend()
        self._calc_std_delta_Rrs()
        self._calc_normalised_difference_Rrs()
        self._calc_power_spectrum()

        return self.Syy

class helper_methods():
    @staticmethod
    def read_iop_from_file(file_name):
        """
        Generic IOP reader that interpolates the iop to the common wavelengths defined in the constructor
        :param file_name: filename and path of the csv file
        :returns interpolated iop
        """

        wave = []
        iop = []

        lg.info('Reading :: ' + file_name)
        if os.path.isfile(file_name):
            iop_reader = csv.reader(open(file_name), delimiter=',', quotechar='"')
            wave = iop_reader.next()
            for row in iop_reader:
                iop.append(row)
        else:
            lg.exception('Problem reading file :: ' + file_name)
            raise IOError

        return scipy.asarray(wave, dtype=float), scipy.asarray(iop, dtype=float)

