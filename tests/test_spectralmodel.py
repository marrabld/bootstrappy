__author__ = 'marrabld'

import sys
import pylab
import numpy as np

sys.path.append('../..')

import libbootstrap.spectralmodel as spectralmodel
import libbootstrap.spectra_generator as spectra_generator

#test_data_file = '/home/marrabld/Projects/phd/bootstrappy/inputs/test_data/hope_rrs.csv'
test_data_file = '/home/marrabld/Projects/phd/test_dataset/bootstrappy_test/batch_report.csv'
sm = spectralmodel.BuildSpectralModel(test_data_file)
# sm._detrend()
# sm._calc_std_Rrs()
# sm._calc_normalised_difference_Rrs()
# Y = sm._calc_power_spectrum()

Y = sm.build()

# pylab.semilogy(Y)
# pylab.show()

pylab.clf()

sg = spectra_generator.GenerateRealisation(sm, 100)
rn = sg._gen_random_numbers(100)
N = sg._calc_fft_random_numbers()
Y = sg._calc_rand_powerspectrum()
y = sg._calc_inv_rand_powerspectrum()
y_hat = sg._calc_normalised_difference_inv_rand_powerspectrum()
z = sg._calc_z()
Rrs = sg.gen_Rrs(sm.Rrs[0, :])

np.savetxt('/home/marrabld/Projects/phd/bootstrappy/outputs/Rrs.csv', np.vstack((sm.wave, np.real(Rrs))), delimiter=',')

for row in Rrs:
    pylab.plot(sm.wave, row)
pylab.show()

print('OK')