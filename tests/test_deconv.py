__author__ = 'marrabld'

import sys
import scipy

sys.path.append('../..')

import unittest
import libbootstrap.deconv
import pylab


class setUp(unittest.TestCase):
    # wavelengths = scipy.asarray(
    #    [410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620,
    #     630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730])
    wavelengths = scipy.asarray([470., 490., 510., 530., 550., 570., 590., 610., 630., 650., 670., 690., 710., 730.])

    # dc = libbootstrap.deconv.OpticalModel(wavelengths)
    # dc = libbootstrap.deconv.McKeeModel(wavelengths)
    dc = libbootstrap.deconv.McKeeModelCase2(wavelengths)
    #dc = libbootstrap.deconv.BCDeep(wavelengths)

    # dc.bio_optical_parameters.build_a_cdom(1.0, 0.014)  # todo, change these to realistic values
    # dc.bio_optical_parameters.write_iop_to_file(wavelengths, dc.bio_optical_parameters.a_cdom,
    #                                             '../inputs/iop_files/ag.csv')
    #
    # dc.bio_optical_parameters.build_a_cdom(1.0, 0.014)  # todo, change these to realistic values
    # dc.bio_optical_parameters.write_iop_to_file(wavelengths, dc.bio_optical_parameters.a_cdom,
    #                                             '../inputs/iop_files/ad.csv')  ## and look s like cdom anayway
    #
    # dc.bio_optical_parameters.build_bbp(1.0, 1.0)
    # dc.bio_optical_parameters.write_iop_to_file(wavelengths, dc.bio_optical_parameters.b_bp,
    #                                             '../inputs/iop_files/bbm.csv')

    # dc.read_all_iops_from_files()
    # #p = 0.6
    # #dc.bio_optical_parameters.scale_aphi(p)
    # dc.bio_optical_parameters.scale_bphi(0.)
    # dc.bio_optical_parameters.build_a()
    # dc.bio_optical_parameters.write_a_to_file('../inputs/iop_files/a.csv')
    # #
    # dc.bio_optical_parameters.build_bb()
    # dc.bio_optical_parameters.write_bb_to_file('../inputs/iop_files/bb.csv')
    # dc.read_bb_from_file()
    # dc.read_a_from_file()

    # --------------------------------------------------#
    # We model an Rrs for the test
    # --------------------------------------------------#
    rrs_file = '../inputs/iop_files/hope_rrs.csv'
    #rrs = dc.bio_optical_parameters.bb / (dc.bio_optical_parameters.a + dc.bio_optical_parameters.bb)
    #dc.bio_optical_parameters.write_iop_to_file(wavelengths, rrs, rrs_file)

    # --------------------------------------------------#
    # Add some random noise
    # --------------------------------------------------#

    dc.read_rrs_from_file(rrs_file)
    dc.read_all_iops_from_files()

    data = dc.run(num_iters=50)
    #print(iops['x'])
    print(data[0])
    print(data[1])
    #print(data[2])
    #print(data[3])

    check = dc.func(data[0])
    import pylab

    pylab.plot(check, '*')
    pylab.plot(dc.rrs[0, :])
    pylab.show()
    #print(check)

    #check = dc.opt_func([1, 1., 1., 1.], rrs)
    #print(check)

