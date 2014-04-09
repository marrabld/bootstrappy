__author__ = 'marrabld'

import os
import sys

sys.path.append("../..")

import logger as log
import scipy
import scipy.optimize
import libbootstrap
import libbootstrap.state
import csv

DEBUG_LEVEL = libbootstrap.state.State().debug
lg = log.logger
lg.setLevel(DEBUG_LEVEL)


class BioOpticalParameters():
    def __init__(self, wavelengths):
        self.wavelengths = scipy.asarray([wavelengths])
        self.b_bp = scipy.asarray([])
        self.a_cdom = scipy.asarray([])
        self.a_phi = scipy.asarray([])
        self.a_water = scipy.asarray([])
        self.b_water = scipy.asarray([])

        self.b_b = scipy.asarray([])
        self.b = scipy.asarray([])
        self.a = scipy.asarray([])
        self.c = scipy.asarray([])
        self.rrs = scipy.asarray([])


    def build_bbp(self, x, y, wave_const=550):
        r"""
        Builds the particle backscattering function  :math:`X(\frac{550}{\lambda})^Y`
        param: x function coefficient
        param: y order of the power function
        param: waveConst wave constant Default 550 nm
        retval: null
        """
        lg.info('Building b_bp spectra')
        self.b_bp = x * (wave_const / self.wavelengths) ** y

    def build_a_cdom(self, g, s, wave_const=400):
        r"""
        Builds the CDOM absorption function :: :math:`G \exp (-S(\lambda - 400))`
        param: g function coefficient
        param: s slope factor
        param: wave constant
        retval null
        """
        lg.info('building CDOM absorption')
        self.a_cdom = g * scipy.exp(-s * (self.wavelengths - wave_const))

    def read_aphi_from_file(self, file_name):
        """

        """
        lg.info('Reading ahpi absorption')
        try:
            self.a_phi = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.a_phi = -1

        return self.a_phi

    def read_b_from_file(self, file_name):
        """

        """
        lg.info('Reading total scattering')
        try:
            self.b = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b = -1

        return self.b

    def read_bb_from_file(self, file_name):
        """

        """
        lg.info('Reading backscattering')
        try:
            self.bb = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.bb = -1

        return self.bb

    def read_a_from_file(self, file_name):
        """

        """
        lg.info('Reading total absorption')
        try:
            self.a = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.a = -1

        return self.a

    def scale_aphi(self, scale_paraemter):
        """

        """
        lg.info('Scaling a_phi by :: ' + str(scale_paraemter))
        try:
            self.a_phi = self.a_phi * scale_paraemter
        except:
            lg.exception("Can't scale a_phi, check that it has been defined ")

    def read_pure_water_absorption_from_file(self, file_name):
        """

        """
        lg.info('Reading water absorption from file')
        try:
            self.a_water = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.a_phi = -1

        return self.a_water

    def read_pure_water_scattering_from_file(self, file_name):
        """

        """
        lg.info('Reading water scattering from file')
        try:
            self.b_water = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b_phi = -1

        return self.b_water

    def read_sub_surface_reflectance_from_file(self, file_name):
        """

        """
        lg.info('Reading subsurface reflectance from file')
        try:
            self.rrs = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.rrs = -1

        return self.rrs


    def _read_iop_from_file(self, file_name):
        """
        Generic IOP reader that interpolates the iop to the common wavelengths defined in the constructor

        returns: interpolated iop
        """
        lg.info('Reading :: ' + file_name + ' :: and interpolating to ' + str(self.wavelengths))

        if os.path.isfile(file_name):
            iop_reader = csv.reader(open(file_name), delimiter=',', quotechar='"')
            wave = iop_reader.next()
            iop = iop_reader.next()
        else:
            lg.exception('Problem reading file :: ' + file_name)
            raise IOError

        try:
            return scipy.interp(self.wavelengths, wave, iop)
        except IOError:
            lg.exception('Error interpolating IOP to common wavelength')
            return -1

    def write_b_to_file(self, file_name):
        self._write_iop_to_file(self.b, file_name)

    def write_c_to_file(self, file_name):
        self._write_iop_to_file(self.c, file_name)

    def _write_iop_to_file(self, iop, file_name):
        lg.info('Writing :: ' + file_name)
        f = open(file_name, 'w')
        for i in scipy.nditer(iop):
            f.write(str(i) + '\n')

    def build_bb(self):
        lg.info('Building bb spectra')
        self.b_b = self.b_bp + self.b_water

    def build_b(self, scattering_fraction=0.2):
        lg.info('Building b with scattering fraction of :: ' + str(scattering_fraction))
        self.b = self.b_b / scattering_fraction

    def build_a(self):
        lg.info('Building total absorption')
        self.a = self.a_water + self.a_cdom + self.a_phi

    def build_c(self):
        lg.info('Building total attenuation C')
        self.c = self.a + self.b

    def build_all_iop(self):
        lg.info('Building all b and c from IOPs')

        self.build_a()
        self.build_bb()
        self.build_b()
        self.build_c()


class Model():
    def __init__(self):
        self.bb = None
        self.a = None
        self.bio_optical_parameters = BioOpticalParameters()
        self.aw = None
        self.bw = None
        self.rrs = None
        self.bb_phi = None
        self.bb_m = None
        self.bb_d = None
        self.a_phi = None
        self.a_m = None
        self.a_d = None
        self.a_g = None

    def read_bb_from_file(self, filename='../inputs/iop_files/bb.csv'):
        self.bb = self.bio_optical_parameters.read_bb_from_file(filename)

    def read_a_from_file(self, filename='../inputs/iop_files/a.csv'):
        self.a = self.bio_optical_parameters.read_a_from_file(filename)

    def read_bw_from_file(self, filename='../inputs/iop_files/b_water.csv'):
        self.bw = self.bio_optical_parameters.read_pure_water_absorption_from_file(filename)

    def read_aw_from_file(self, filename='../inputs/iop_files/a_water.csv'):
        self.aw = self.bio_optical_parameters.read_pure_water_absorption_from_file(filename)

    def read_rrs_from_file(self, filename='../inputs/iop_files/rrs.csv'):
        self.rrs = self.bio_optical_parameters.read_sub_surface_reflectance_from_file(filename)

    def opt_func(self, ydata, phi, m, d, g):
        Bb = (phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        return ydata - (Bb / A)  #  Residual

    def solve_opt_func(self, ydata):
        guess = {}
        opt_data = scipy.zeros_like(ydata)
        guess['phi'] = 1  # todo, change this to kwags and set default values.
        guess['m'] = 1
        guess['d'] = 1
        guess['g'] = 1

        for i_iter, row in enumerate(scipy.nditer(ydata)):
            opt_data[i_iter] = scipy.optimize.leastsq(self.opt_func(),
                                                      args=(row, guess['phi'], guess['m'], guess['d'], guess['g']))

        return opt_data


    def run(self):
        #--------------------------------------------------#
        #  Todo : check to see if the inputs are not none
        #--------------------------------------------------#
        outputfile = 'bb_on_a.csv'
        self.read_bb_from_file()
        self.read_a_from_file()
        self.read_bw_from_file()
        self.read_aw_from_file()
        self.read_rrs_from_file()

        data = self.solve_opt_func(self.rrs)

        data.tofile(outputfile)




