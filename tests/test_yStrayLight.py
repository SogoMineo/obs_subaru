#
# LSST Data Management System
#
# Copyright 2008-2016  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import unittest

import lsst.utils.tests
import lsst.daf.base as dafBase

from lsst.obs.subaru.strayLight import waveletCompression
from lsst.obs.subaru.strayLight import yStrayLight
from lsst.obs.subaru.strayLight import rotatorAngle

import numpy


class CDF97TestCase(lsst.utils.tests.TestCase):
    """
    Test cases for wavelet transformation.
    """

    # A random array that will be passed to wavelet transformation.
    # In (a[n]): a[0] = len(__randomData), a[i+1] = a[i] // 2,
    # there should be both even numbers and odd numbers for code coverage.
    # We test two (a[n]):
    #   (a[n]) = 13, 6, 3, 1
    #   (a[n]) = 10, 5, 2, 1
    __randomData1 = numpy.array([
        0.33410047828409528, 0.49744873124074418, 0.40334751942075375,
        0.23242598684572702, 0.29937712479697964, 0.49308287527627470,
        0.57076777081529306, 0.72426886638248378, 0.87743578755773366,
        0.38771114763968961, 0.93160863980449626, 0.71077077724187865,
        0.21539110804957817,
    ], dtype=float)

    __randomData2 = numpy.array([
        0.74527547327726684, 0.67385893766408611, 0.98665305680943483,
        0.26878725917354596, 0.96389866239451683, 0.29879804256075682,
        0.28861891121374217, 0.15048978031694449, 0.80020182616359903,
        0.11966611228991342,
    ], dtype=float)

    # Correct answer of cdf_9_7(randomData1)
    __cdf97_randomData1 = numpy.array([
        1.9593790280735301, 0.52485393596830365, -0.28928523014496271,
        -0.33963009860713367, 0.051579598906558256, 0.079250812940339505,
        0.2383573500355872, 0.099564745865735466, -0.092444164913274493,
        0.068720864526320313, 0.0092003156974255554, -0.45829594549676755,
        0.15293791864108841,
    ], dtype=float)

    def testCDF97Roundtrip(self):
        """
        Test roundtrip: icdf_9_7(cdf_9_7(x)) == x
        """
        array2d = self.__randomData1.reshape(-1, 1) - self.__randomData2.reshape(1, -1)
        height, width = array2d.shape

        for iLevel in range(5):
            for jLevel in range(5):
                # We want to try two types of args (int and tuple)
                # for the param `level`
                level = iLevel if iLevel == jLevel else (iLevel, jLevel)

                temp = waveletCompression.cdf_9_7(numpy.copy(array2d), level)
                temp = waveletCompression.icdf_9_7(temp, level)
                self.assertFloatsAlmostEqual(
                    temp,
                    array2d,
                    rtol = None,
                    atol = 2e-15,
                )

    def testCDF97(self):
        """
        Test cdf_9_7(x)
        """
        self.assertFloatsAlmostEqual(
            waveletCompression.cdf_9_7(numpy.copy(self.__randomData1), level=5),
            self.__cdf97_randomData1,
        )
        self.assertFloatsEqual(
            waveletCompression.cdf_9_7(numpy.copy(self.__randomData1), level=0),
            self.__randomData1,
        )

    def testScaledSize(self):
        originalSize = size = 35191 # random value
        for level in range(20):
            self.assertEqual(
                size,
                waveletCompression.scaled_size(originalSize, level),
            )
            size = (size + 1) // 2


class PeriodicCDF97TestCase(lsst.utils.tests.TestCase):
    """
    Test cases for wavelet transformation (periodic boundary).
    """

    # A random array that will be passed to wavelet transformation.
    # Its length must be 2**n
    __randomData = numpy.array([
        0.33410047828409528, 0.49744873124074418,
        0.40334751942075375, 0.23242598684572702,
    ], dtype=float)

    # Correct answer of periodic_cdf_9_7(randomData)
    __cdf97_randomData = numpy.array([
        0.73366135789566034, 0.056290439589035478,
        0.11258937342364359, -0.11794449079762188
    ], dtype=float)

    def testPeriodicCDF97Roundtrip(self):
        """
        Test roundtrip: periodic_icdf_9_7_1d(periodic_cdf_9_7_1d(x)) == x
        """
        arr = self.__randomData

        for level in range(2):
            temp = waveletCompression.periodic_cdf_9_7_1d(numpy.copy(arr), level, axis=0)
            temp = waveletCompression.periodic_icdf_9_7_1d(temp, level, axis=0)

            self.assertFloatsAlmostEqual(
                temp,
                arr,
                rtol = None,
                atol = 2e-16,
            )

        with self.assertRaises(ValueError):
            # Must raise error if len(data) is an odd number
            waveletCompression.periodic_cdf_9_7_1d(numpy.copy(arr[:3]), 1, axis=0)

        with self.assertRaises(ValueError):
            # Must raise error if len(data) is an odd number
            waveletCompression.periodic_icdf_9_7_1d(numpy.copy(arr[:3]), 1, axis=0)

    def testPeriodicCDF97(self):
        """
        Test periodic_cdf_9_7_1d(x)
        """
        self.assertFloatsAlmostEqual(
            waveletCompression.periodic_cdf_9_7_1d(numpy.copy(self.__randomData), 2, axis=0),
            self.__cdf97_randomData,
        )
        self.assertFloatsEqual(
            waveletCompression.periodic_cdf_9_7_1d(numpy.copy(self.__randomData), 0, axis=0),
            self.__randomData,
        )


class RotatorAngleTestCase(lsst.utils.tests.TestCase):
    """
    Test cases for inrStartEnd():
    the function to calculate instrument rotator angle.
    """
    # These test data were picked randomly from the public data release 1.
    # Ideally, we should have some data with INST-PA != 0, but such data were
    # not found in the public data release 1.
    # The values in 'inr in header' field were obtained from actual FITS headers.
    # The values in 'inr-str' and 'inr-end' fields were return values from inrStartEnd(),
    # which is to be tested.
    __testData = [
        # crval1 ,  crval2,inst_pa,  mjd-start ,  mjd-end   , inr in header, inr-str,     inr-end
        (339.6925, -0.9347, -360.0, 57224.56207, 57224.56440,  15.496024 ,  15.5042018 ,  17.5519814 ),
        (347.7592, -0.4000,    0.0, 57309.24328, 57309.24504, -56.436975 , -56.4435984 , -55.9902197 ),
        (214.3979, -0.0811,    0.0, 56744.57128, 56744.57362,  45.268328 ,  45.2859895 ,  46.3362063 ),
        (331.2959, -0.8242,  360.0, 57243.52133, 57243.52367,  40.233028 ,  40.2381203 ,  41.4696707 ),
        ( 31.0039, -4.3097,    0.0, 57038.31261, 57038.31495,  57.823085 ,  57.8361065 ,  58.3066506 ),
        (134.9843,  0.0975,    0.0, 57038.37301, 57038.37534, -61.183082 , -61.1927895 , -60.7747956 ),
        (140.0230,  0.4153,    0.0, 57341.61755, 57341.61989, -42.545777 , -42.5574953 , -41.3109364 ),
        (340.9412,  0.2217,    0.0, 57218.55889, 57218.56065,  -6.5420110,  -6.54017916,  -4.77026297),
        (218.2380, -1.3000,    0.0, 57159.35315, 57159.35490, -28.561150 , -28.5520606 , -27.2905344 ),
        (243.8357, 42.7644,    0.0, 57159.59252, 57159.59428, 103.57883  , 103.574821  , 103.043506  ),
    ]

    def testInrStartEnd(self):
        for crval1, crval2, instpa, mjd_str, mjd_end, inr_in_header, inr_str, inr_end in self.__testData:
            header = self.buildPropertySet({
                'CRVAL1' : crval1 ,
                'CRVAL2' : crval2 ,
                'INST-PA': instpa ,
                'MJD-STR': mjd_str,
                'MJD-END': mjd_end,
            })

            start, end = rotatorAngle.inrStartEnd(header)
            start, end = self.modulo(360, start, end)

            inr_str, inr_end = self.modulo(360, inr_str, inr_end)

            self.assertFloatsAlmostEqual(
                start,
                inr_str,
                rtol = 2e-7,
                atol = None,
            )
            self.assertFloatsAlmostEqual(
                end,
                inr_end,
                rtol = 2e-7,
                atol = None,
            )

    @staticmethod
    def buildPropertySet(dictlike):
        pset = dafBase.PropertySet()
        for k, v in dictlike.items():
            pset.add(k, v)
        return pset

    @staticmethod
    def modulo(m, a, *others):
        """
        modulo(m, a) = a % m,
        modulo(m, a, b, c, ...) = (a % m, b', c', ...),
        where
            b' = b - floor(a / m) * m,
            c' = c - floor(a / m) * m,
            ...
        """
        diff = numpy.floor(float(a) / m) * m
        ret = [a - diff] + [b - diff for b in others]
        return tuple(ret)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
