__author__ = 'marrabld'

import sys
import seaborn as sns

import numpy as np
import scipy

# !mkdir tmp
import pylab
import sys

sys.path.append("..")

import libbootstrap
import libbootstrap.deconv

rrs_file = '/home/marrabld/Projects/phd/test_dataset/bootstrappy_test/1_batch_report.csv'
# np.savetxt(rrs_file, np.vstack((sm.wave, np.real(sub_Rrs))), delimiter=',');

wavelengths = scipy.asarray([410.0, 430.0, 450.0, 470.0, 490.0, 510.0, 530.0, 550.0, 570.0, 590.0, 610.0, 630.0, 650.0, 670.0, 690.0, 710.0, 730.0])
dc = libbootstrap.deconv.HopeDeep(wavelengths)
#deep = libbootstrap.deconv.BCDeep(wavelengths)
#dc = libbootstrap.deconv.McKeeModel(wavelengths)
qaa = libbootstrap.deconv.QAA(wavelengths)
deep = libbootstrap.deconv.HopeDeep(wavelengths)

deep.read_rrs_from_file(rrs_file)
deep.read_all_iops_from_files()

dc.read_rrs_from_file(rrs_file)
dc.read_all_iops_from_files()

qaa.read_rrs_from_file(rrs_file)
qaa.read_all_iops_from_files()

data = dc.run(num_iters=1)
qaa_data = qaa.run(num_iters=1)

print(data[0])
print(qaa_data)

check = deep.func(data[0])
check_qaa = deep.func(qaa_data)
forward = deep.func([0.01, 0.1, 0.0, 0.1])  # P, m, d, G --> [0.01, 0.1, 0.0, 0.1]

print(dc.rrs.shape)
pylab.plot(wavelengths, dc.rrs[0, :], alpha=0.75, label='Planarrad')

pylab.plot(wavelengths, check, '--o', alpha=0.75, label='Forward - Inverted deep HOPE')
pylab.plot(wavelengths, forward, '--o', alpha=0.75, label='Forward - Planarrad Inputs')
pylab.plot(wavelengths, check_qaa, '--o', alpha=0.75, label='Forward - QAA derived params passed to deep HOPE')

pylab.xlabel('Wavelength (nm)')
pylab.ylabel(r'$R_{rs}$ $(sr^{-1})$')
pylab.legend()

pylab.show()