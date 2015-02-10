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
        # self.b_bp = None #scipy.asarray([])
        self.b_bm = None
        self.a_cdom = None  # scipy.asarray([])
        self.a_phi = None  # scipy.asarray([])
        self.a_water = None  # scipy.asarray([])
        self.b_bwater = None  # scipy.asarray([])

        self.b_b = None  # scipy.asarray([])
        self.b_bphi = None
        # self.b = None #scipy.asarray([])
        self.a = None  #scipy.asarray([])
        self.c = None  #scipy.asarray([])
        self.rrs = None  #scipy.asarray([])


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

    def read_b_bphi_from_file(self, file_name):
        """

        """
        lg.info('Reading ahpi absorption')
        try:
            self.b_bphi = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b_bphi = -1

        return self.b_bphi

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

    def read_bbm_from_file(self, file_name):
        """

        """
        lg.info('Reading particle backscattering')
        try:
            self.b_bm = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b_bm = -1

        return self.b_bm

    def read_bbd_from_file(self, file_name):
        """

        """
        lg.info('Reading particle backscattering')
        try:
            self.b_bd = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b_bd = -1

        return self.b_bd

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

    def read_am_from_file(self, file_name):
        """

        """
        lg.info('Reading total absorption')
        try:
            self.am = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.am = -1

        return self.am

    def read_ad_from_file(self, file_name):
        """

        """
        lg.info('Reading total absorption')
        try:
            self.ad = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.ad = -1

        return self.ad

    def read_ag_from_file(self, file_name):
        """

        """
        lg.info('Reading total absorption')
        try:
            self.ag = self._read_iop_from_file(file_name)
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.ag = -1

        return self.ag

    def scale_aphi(self, scale_paraemter):
        """

        """
        lg.info('Scaling a_phi by :: ' + str(scale_paraemter))
        try:
            self.a_phi = self.a_phi * scale_paraemter
        except:
            lg.exception("Can't scale a_phi, check that it has been defined ")

    def scale_bphi(self, scale_paraemter):
        """

        """
        lg.info('Scaling b_phi by :: ' + str(scale_paraemter))
        try:
            self.b_bphi = self.b_bphi * scale_paraemter
        except:
            lg.exception("Can't scale b_phi, check that it has been defined ")

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

    def read_pure_water_scattering_from_file(self, file_name, scattering_factor=0.2):
        """

        """
        lg.info('Reading water scattering from file')
        try:
            self.b_bwater = self._read_iop_from_file(file_name) * scattering_factor
        except:
            lg.exception('Problem reading file :: ' + file_name)
            self.b_bwater = -1

        return self.b_bwater

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
                # r = scipy.interp(self.wavelengths[0, :], wave, iop[i_iter, :])
                int_iop[i_iter, :] = scipy.interp(self.wavelengths, wave, iop[i_iter, :])
            return int_iop
        except IOError:
            lg.exception('Error interpolating IOP to common wavelength')
            return -1

    # def write_b_to_file(self, file_name):
    # self.write_iop_to_file(self.wavelengths, self.b, file_name)

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
        self.b_b = self.b_bphi + self.b_bm + self.b_bd + (self.b_bwater)
        #self.b_b = self.b_b * 0.2
        #phi * self.bb_phi + m * self.bb_m + d * self.bb_d + self.bw

    # def build_b(self, scattering_fraction=0.2):
    #     lg.info('Building b with scattering fraction of :: ' + str(scattering_fraction))
    #     self.b = self.b_b / scattering_fraction

    def build_a(self):
        lg.info('Building total absorption')
        self.a = self.a_water + self.ag + self.a_phi + self.ad + self.am

    def build_c(self):
        lg.info('Building total attenuation C')
        self.c = self.a + self.b

    def build_all_iop(self):
        lg.info('Building all b and c from IOPs')

        self.build_a()
        self.build_bb()
        #self.build_b()
        #self.build_c()


class OpticalModel():
    def __init__(self, wavelengths):
        self.bb = None
        self.a = None
        self.bio_optical_parameters = BioOpticalParameters(wavelengths)
        self.aw = None
        self.bw = None
        self.rrs = None
        self.b_bphi = None
        self.bb_m = None
        self.bb_d = None
        self.a_phi = None
        self.a_m = None
        self.a_d = None
        self.a_g = None
        self.b_bw = None

    def read_bb_from_file(self, filename='../inputs/iop_files/bb.csv'):
        self.bb = self.bio_optical_parameters.read_bb_from_file(filename)

    def read_a_from_file(self, filename='../inputs/iop_files/a.csv'):
        self.a = self.bio_optical_parameters.read_a_from_file(filename)

    def read_bw_from_file(self, filename='../inputs/iop_files/b_water.csv'):
        self.b_bw = self.bio_optical_parameters.read_pure_water_scattering_from_file(filename)

    def read_aw_from_file(self, filename='../inputs/iop_files/a_water.csv'):
        self.aw = self.bio_optical_parameters.read_pure_water_absorption_from_file(filename)

    def read_rrs_from_file(self, filename='../inputs/iop_files/rrs.csv'):
        self.rrs = self.bio_optical_parameters.read_sub_surface_reflectance_from_file(filename)

    def read_aphi_from_file(self, filename='../inputs/iop_files/aphi.csv'):
        self.a_phi = self.bio_optical_parameters.read_aphi_from_file(filename)

    def read_bbm_from_file(self, filename='../inputs/iop_files/bbm.csv'):
        self.b_bm = self.bio_optical_parameters.read_bbm_from_file(filename)

    def read_bbd_from_file(self, filename='../inputs/iop_files/bbd.csv'):
        self.b_bd = self.bio_optical_parameters.read_bbd_from_file(filename)

    def read_am_from_file(self, filename='../inputs/iop_files/am.csv'):
        self.a_m = self.bio_optical_parameters.read_am_from_file(filename)

    def read_ad_from_file(self, filename='../inputs/iop_files/ad.csv'):
        self.a_d = self.bio_optical_parameters.read_ad_from_file(filename)

    def read_ag_from_file(self, filename='../inputs/iop_files/ag.csv'):
        self.a_g = self.bio_optical_parameters.read_ag_from_file(filename)

    def read_bbphi_from_file(self, filename='../inputs/iop_files/b_bphi.csv'):
        self.b_bphi = self.bio_optical_parameters.read_b_bphi_from_file(filename)

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
                                                 '../inputs/iop_files/b_bphi.csv']):
        self.read_bb_from_file(filename=filelist[0])
        self.read_a_from_file(filename=filelist[1])
        self.read_bw_from_file(filename=filelist[2])
        self.read_aw_from_file(filename=filelist[3])
        self.read_rrs_from_file(filename=filelist[4])
        self.read_aphi_from_file(filename=filelist[5])
        self.read_bbm_from_file(filename=filelist[6])
        self.read_bbd_from_file(filename=filelist[7])
        self.read_am_from_file(filename=filelist[8])
        self.read_ad_from_file(filename=filelist[9])
        self.read_ag_from_file(filename=filelist[10])
        self.read_bbphi_from_file(filename=filelist[11])

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.b_bphi + m * self.b_bm + d * self.b_bd + self.b_bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        return scipy.squeeze(Bb / A)


    def opt_func(self, args, *params):
        ydata = params[0]
        return_vals = self.func(args)
        res = scipy.squeeze(ydata - return_vals)
        return (res ** 2).sum()

        # return scipy.squeeze(ydata - return_vals)  #  Residual

    def solve_opt_func(self, ydata, **kwargs):
        opt_data = scipy.zeros((ydata.shape[0], 4))
        # phi = kwargs.get('phi', 0.01)
        # m = kwargs.get('m', 0.01)
        # d = kwargs.get('d', 0.01)
        # g = kwargs.get('g', 0.1)

        # P0 = [phi, m, d, g]

        for i_iter in range(0, ydata.shape[0]):
            #opt_data[i_iter, :], cov_x = scipy.optimize.leastsq(self.opt_func, P0, args=ydata[i_iter, :], full_output=0)
            #_args = tuple(map(tuple, ydata[i_iter, :]))
            _args = tuple([tuple(row) for row in ydata])
            #opt_data[i_iter, :], cov_x = scipy.optimize.minimize(self.opt_func, P0, args=_args)
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='Nelder-Mead')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='Powell')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='CG')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='BFGS')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='Newton-CG') jac needed
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='L-BFGS-B')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='COBYLA')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='SLSQP')
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='dogleg') # jac required
            #opt_data = scipy.optimize.minimize(self.opt_func, P0, args=_args, method='trust-ncg') # jac required

            opt_data = scipy.optimize.brute(self.opt_func, ranges=((0., 0.1), (0., 1.), (0., 0.2), (0., 0.2)), Ns=10,
                                            full_output=True, args=_args, finish=scipy.optimize.fmin)
            #opt_data = scipy.optimize.brute(self.opt_func, ranges=((0.5, 2.), (0.5, 2.), (0.5, 2.), (0.5, 2.)), Ns=16, full_output=True, args=_args, finish=None)

        return opt_data

    def run(self, outputfile='results.csv', **kwargs):
        # --------------------------------------------------#
        #  Todo : check to see if the inputs are not none
        #--------------------------------------------------#
        import inspect
        this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        data_list = []
        residual_list = []
        #outputfile = 'bb_on_a.csv'
        # print(os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'bb.csv')))
        bb_file = os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'bb.csv'))
        self.read_bb_from_file(bb_file)
        a_file = os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'a.csv'))
        self.read_a_from_file(a_file)
        b_water_file = os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'b_water.csv'))
        self.read_bw_from_file(b_water_file)
        a_water_file = os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'a_water.csv'))
        self.read_aw_from_file(a_water_file)
        rrs_file = os.path.abspath(os.path.join(this_dir, '../inputs/iop_files', 'rrs.csv'))
        self.read_rrs_from_file(rrs_file)

        num_iters = kwargs.get('num_iters', 10)
        noise_magnitude = kwargs.get('noise_magnitude', 0.005)

        for i_iter in range(0, num_iters):
            self.noise = scipy.random.normal(0, noise_magnitude, self.rrs.shape[1])
            data = self.solve_opt_func(self.rrs, **kwargs)

        data_list.append(data)
        residual_list.append(data[1])
        idx = residual_list.index(min(residual_list))

        #--------------------------------------------------#
        # Do a forward model with the inverted parameters
        #--------------------------------------------------#

        # data.tofile(outputfile)
        #
        # with open(outputfile, 'w') as fp:
        #     file_writer = csv.writer(fp, delimiter=',')
        #     for row in data:
        #         file_writer.writerow(row)

        return data_list[idx]


class McKeeModel(OpticalModel):
    def __init__(self, wavelengths):
        OpticalModel.__init__(self, wavelengths)

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.b_bphi + m * self.b_bm + d * self.b_bd + self.bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        return scipy.squeeze(Bb / A)


    def opt_func(self, params, ydata):
        return_vals = self.func(params)

        return scipy.squeeze(ydata - return_vals)  # Residual


class McKeeModelCase2(OpticalModel):
    def __init__(self, wavelengths):
        OpticalModel.__init__(self, wavelengths)

    def func(self, params):
        phi = params[0]
        m = params[1]
        d = params[2]
        g = params[3]

        Bb = (phi * self.b_bphi + m * self.b_bm + d * self.b_bd + self.b_bw)
        A = (phi * self.a_phi + m * self.a_m + d * self.a_d + g * self.a_g + self.aw)

        rrs = Bb / (A + Bb) + self.noise
        # Rrs = (0.5 * rrs) / (1 - 1.5 * rrs)

        return scipy.squeeze(rrs)


    def opt_func(self, args, *params):
        ydata = params[0]
        return_vals = self.func(args)
        res = scipy.squeeze(ydata - return_vals)
        return (res ** 2).sum()

        # return scipy.squeeze(ydata - return_vals)  # Residual


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



