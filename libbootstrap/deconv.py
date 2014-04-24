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
import pylab

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


    def build_bbp(self, x, y, wave_const=550.0):
        r"""
        Builds the particle backscattering function  :math:`X(\frac{550}{\lambda})^Y`
        param: x function coefficient
        param: y order of the power function
        param: waveConst wave constant Default 550 nm
        retval: null
        """
        lg.info('Building b_bp spectra')
        self.b_bp = x * (wave_const / self.wavelengths) ** y
        return self.b_bp

    def build_a_cdom(self, g, s, wave_const=400.0):
        r"""
        Builds the CDOM absorption function :: :math:`G \exp (-S(\lambda - 400))`
        param: g function coefficient
        param: s slope factor
        param: wave constant
        retval null
        """
        lg.info('building CDOM absorption')
        self.a_cdom = g * scipy.exp(-s * (self.wavelengths - wave_const))
        return self.a_cdom

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
            wave = scipy.float32(iop_reader.next())
            iop = scipy.zeros_like(wave)
            for row in iop_reader:
                iop = scipy.vstack((iop, row))

            iop = scipy.float32(iop[1:, :])  # drop the first row of zeros
        else:
            lg.exception('Problem reading file :: ' + file_name)
            raise IOError

        try:
            int_iop = scipy.zeros((iop.shape[0], self.wavelengths.shape[1]))
            for i_iter in range(0, iop.shape[0]):
                #r = scipy.interp(self.wavelengths[0, :], wave, iop[i_iter, :])
                int_iop[i_iter, :] = scipy.interp(self.wavelengths, wave, iop[i_iter, :])
            return int_iop
        except IOError:
            lg.exception('Error interpolating IOP to common wavelength')
            return -1

    def write_b_to_file(self, file_name):
        self.write_iop_to_file(self.wavelengths, self.b, file_name)

    def write_c_to_file(self, file_name):
        self.write_iop_to_file(self.wavelengths, self.c, file_name)

    def write_a_to_file(self, file_name):
        self.write_iop_to_file(self.wavelengths, self.a, file_name)

    def write_bb_to_file(self, file_name):
        self.write_iop_to_file(self.wavelengths, self.b_b, file_name)

    def write_iop_to_file(self, wavelengths, iop, file_name):
        lg.info('Writing :: ' + file_name)
        f = open(file_name, 'w')
        for i, wave in enumerate(scipy.nditer(wavelengths)):
            if i < self.wavelengths.shape[1] - 1:
                f.write(str(wave) + ',')
            else:
                f.write(str(wave))
        f.write('\n')

        for i, _iop in enumerate(scipy.nditer(iop)):
            if i < iop.shape[1] - 1:
                f.write(str(_iop) + ',')
            else:
                f.write(str(_iop))

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


class OpticalModel():
    def __init__(self, wavelengths):
        self.bb = None
        self.a = None
        self.bio_optical_parameters = BioOpticalParameters(wavelengths)
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
        self.bw = self.bio_optical_parameters.read_pure_water_scattering_from_file(filename)

    def read_aw_from_file(self, filename='../inputs/iop_files/a_water.csv'):
        self.aw = self.bio_optical_parameters.read_pure_water_absorption_from_file(filename)

    def read_rrs_from_file(self, filename='../inputs/iop_files/rrs.csv'):
        self.rrs = self.bio_optical_parameters.read_sub_surface_reflectance_from_file(filename)

    def read_aphi_from_file(self, filename='../inputs/iop_files/aphi.csv'):
        self.a_phi = self.bio_optical_parameters.read_aphi_from_file(filename)

    def read_bbm_from_file(self, filename='../inputs/iop_files/bbm.csv'):
        self.bb_m = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_bbd_from_file(self, filename='../inputs/iop_files/bbd.csv'):
        self.bb_d = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_am_from_file(self, filename='../inputs/iop_files/am.csv'):
        self.a_m = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_ad_from_file(self, filename='../inputs/iop_files/ad.csv'):
        self.a_d = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_ag_from_file(self, filename='../inputs/iop_files/ag.csv'):
        self.a_g = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_bbphi_from_file(self, filename='../inputs/iop_files/bb_phi.csv'):
        self.bb_phi = self.bio_optical_parameters._read_iop_from_file(filename)

    def read_all_iops_from_files(self, filelist=['../inputs/iop_files/bb.csv',
                                                 '../inputs/iop_files/a.csv',
                                                 '../inputs/iop_files/b_water.csv',
                                                 '../inputs/iop_files/a_water.csv',
                                                 '../inputs/iop_files/rrs.csv',
                                                 '../inputs/iop_files/a_phi.csv',
                                                 '../inputs/iop_files/bbm.csv',
                                                 '../inputs/iop_files/bbd.csv',
                                                 '../inputs/iop_files/am.csv',
                                                 '../inputs/iop_files/ad.csv',
                                                 '../inputs/iop_files/ag.csv',
                                                 '../inputs/iop_files/bb_phi.csv']):
        self.read_bb_from_file(filelist[0])
        self.read_a_from_file(filelist[1])
        self.read_bw_from_file(filelist[2])
        self.read_aw_from_file(filelist[3])
        self.read_rrs_from_file(filelist[4])
        self.read_aphi_from_file(filelist[5])
        self.read_bbm_from_file(filelist[6])
        self.read_bbd_from_file(filelist[7])
        self.read_am_from_file(filelist[8])
        self.read_ad_from_file(filelist[9])
        self.read_ag_from_file(filelist[10])
        self.read_bbphi_from_file(filelist[11])

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        return scipy.squeeze(Bb / A)


    def opt_func(self, params, ydata):
        return_vals = self.func(params)

        return scipy.squeeze(ydata - return_vals)  #  Residual

    def solve_opt_func(self, ydata, **kwargs):
        opt_data = scipy.zeros((ydata.shape[0], 4))
        phi = kwargs.get('phi', 0.01)
        m = kwargs.get('m', 0.01)
        d = kwargs.get('d', 0.01)
        g = kwargs.get('g', 0.1)

        P0 = [phi, m, d, g]

        for i_iter in range(0, ydata.shape[0]):
            opt_data[i_iter, :], cov_x = scipy.optimize.leastsq(self.opt_func, P0, args=ydata[i_iter, :], full_output=0)

        return opt_data

    def run(self, outputfile='results.csv', **kwargs):
        #--------------------------------------------------#
        #  Todo : check to see if the inputs are not none
        #--------------------------------------------------#
        #outputfile = 'bb_on_a.csv'
        self.read_bb_from_file()
        self.read_a_from_file()
        self.read_bw_from_file()
        self.read_aw_from_file()
        self.read_rrs_from_file()

        data = self.solve_opt_func(self.rrs, **kwargs)

        #--------------------------------------------------#
        # Do a forward model with the inverted parameters
        #--------------------------------------------------#

        #data.tofile(outputfile)

        with open(outputfile, 'w') as fp:
            file_writer = csv.writer(fp, delimiter=',')
            for row in data:
                file_writer.writerow(row)

        return data


class McKeeModel(OpticalModel):
    def __init__(self, wavelengths):
        OpticalModel.__init__(self, wavelengths)

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        return scipy.squeeze(Bb / A)


    def opt_func(self, params, ydata):
        return_vals = self.func(params)

        return scipy.squeeze(ydata - return_vals)  #  Residual


class McKeeModelCase2(OpticalModel):
    def __init__(self, wavelengths):
        OpticalModel.__init__(self, wavelengths)

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        rrs = Bb / (A + Bb)
        Rrs = (0.5 * rrs) / (1 - 1.5 * rrs)

        return scipy.squeeze(Rrs)


    def opt_func(self, params, ydata):
        return_vals = self.func(params)

        return scipy.squeeze(ydata - return_vals)  # Residual


class BCDeep(OpticalModel):
    def __init__(self, wavelengths):
        OpticalModel.__init__(self, wavelengths)

    def func(self, params):
        G0_w = 0.0624
        G1_w = 0.0524
        G0_p = 0.0434
        G1_p = 0.1406

        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        k = A + Bb

        Rrs = (G0_w + G1_w * (self.bw / k)) * self.bw / k + (G0_p + G1_p * (self.bb_m / k)) * (self.bb_m / k)

        return scipy.squeeze(Rrs)


    def opt_func(self, params, ydata):
        return_vals = self.func(params)

        return scipy.squeeze(ydata - return_vals)  # Residual


class QAA(OpticalModel):
    def __init__(self, wavelengths):
        """


        :type wavelengths: list
        :param wavelengths:
        """
        OpticalModel.__init__(self, wavelengths)
        self.g0 = 0.0895
        self.g1 = 0.1237
        self.s = 0.015
        self.wavelengths = scipy.asarray(wavelengths)

    def calc_rrs(self, Rrs):
        """

        """
        return Rrs / (0.52 + 1.7 * Rrs)

    def calc_u(self, rrs):
        """

        """
        return (-self.g0 + (self.g0 ** 2 + 4 * self.g1 * rrs) ** (1 / 2)) / 2 * self.g1

    def calc_a555(self, rrs440, rrs555):
        """


        :param a440:
        :param rrs440:
        :param rrs555:
        """

        rho = scipy.log(rrs440 / rrs555)
        a440 = scipy.exp(-2.0 - 1.4 * rho + 0.2 * rho ** 2)
        a555 = 0.0596 + 0.2 * (a440 - 0.01)

        return a555

    def calc_bbp555(self, u555, a555, bbw555):
        """


        :param u555:
        :param a555:
        :param bbw555:
        """
        return ((u555 * a555) / (1.0 - u555)) - bbw555

    def calc_y(self, rrs440, rrs555):
        """


        :param rrs440:
        :param rrs555:
        """
        return 2.2 * (1 - 1.2 * scipy.exp(-0.9 * (rrs440 / rrs555)))

    def calc_bbp(self, bbp555, y):
        """


        :param bbp555:
        :param y:
        """
        return bbp555 * (555.0 / self.wavelengths) ** y

    def calc_a(self, u, bbw, bbp):
        """


        :param u:
        :param bbw:
        :param bbp:
        """
        return ((1 - u) * (bbw + bbp)) / u

    def calc_zeta(self, rrs440, rrs555):
        """

        :param rrs440:
        :param rrs555:
        """
        return 0.71 + (0.06 / (0.8 - (rrs440 / rrs555)))

    def calc_ksi(self):
        """

        """
        return scipy.exp(self.s * (440.0 - 410.0))

    def calc_ag440(self, a410, zeta, a440, ksi, aw410, aw440):
        """

        :param aw440:
        :param aw410:
        :param a410:
        :param zeta:
        :param a440:
        :param ksi:
        """

        val1 = (a410 - zeta * a440) / (ksi - zeta)
        val2 = (aw410 - zeta * aw440) / (ksi - zeta)

        return val1 - val2


    def calc_aphi440(self, a440, ag440, aw440):
        """


        :param a440:
        :param ag440:
        :param aw440:
        """

        return a440 - ag440 - aw440

    def func(self, params):
        """
        For doing the forward model
        """
        aphi440 = params[0]
        ag440 = params[1]
        bbp555 = params[2]
        y = params[3]

        G0_w = 0.0624
        G1_w = 0.0524
        G0_p = 0.0434
        G1_p = 0.1406

        self.read_aw_from_file()
        self.read_bw_from_file()

        aphi = self.a_phi * aphi440
        ag = self.bio_optical_parameters.build_a_cdom(ag440, self.s, wave_const=440)
        bbp = self.bio_optical_parameters.build_bbp(bbp555, y, wave_const=555)

        a = self.aw + aphi + ag
        bb = self.bw + bbp
        u = bb / (a + bb)
        rrs = self.g0 * u + self.g1 * u ** 2
        rrs = 0.5 * rrs / (1 - 1.5 * rrs)

        #k = a + bb

        #Rrs = (G0_w + G1_w * (self.bw / k)) * self.bw / k + (G0_p + G1_p * (self.bb_m / k)) * (self.bb_m / k)
        return scipy.squeeze(rrs)


    def run(self, outputfile='results.csv', **kwargs):
        """

        :param outputfile:
        :param kwargs:
        """

        self.read_rrs_from_file()
        self.read_aw_from_file()
        self.read_bw_from_file()

        params = scipy.zeros((self.rrs.shape[0], 4))

        for i_iter in range(0, self.rrs.shape[0]):
            rrs = self.calc_rrs(self.rrs)[i_iter, :]  #  Todo, do this in a loop

            idx410 = scipy.where(self.wavelengths == 410)[0]
            idx440 = scipy.where(self.wavelengths == 440)[0]
            idx555 = scipy.where(self.wavelengths == 555)[0]

            rrs440 = rrs[idx440]
            rrs555 = rrs[idx555]  # Todo Check for wave 555 in self.wavelengths
            bbw555 = self.bw[0, idx555]
            aw410 = self.aw[0, idx410]
            aw440 = self.aw[0, idx440]

            u = self.calc_u(rrs)
            u555 = u[idx555]

            a555 = self.calc_a555(rrs440, rrs555)
            bbp555 = self.calc_bbp555(u555, a555, bbw555)
            y = self.calc_y(rrs440, rrs555)
            bbp = self.calc_bbp(bbp555, y)
            a = self.calc_a(u, self.bw, bbp)

            a410 = a[0, idx410]
            a440 = a[0, idx440]

            zeta = self.calc_zeta(rrs440, rrs555)
            ksi = self.calc_ksi()
            ag440 = self.calc_ag440(a410, zeta, a440, ksi, aw410, aw440)
            aphi440 = self.calc_aphi440(a440, ag440, aw440)

            params[i_iter, :] = [aphi440, ag440, bbp555, y]

        with open(outputfile, 'w') as fp:
            file_writer = csv.writer(fp, delimiter=',')
            fp.write('aphi440, ag440, bbp555, y \n')
            for row in params:
                file_writer.writerow(row)

        return params


