"""This module is a test suite for the generate_numbers_sequence module."""
import generate_numbers_sequence
import glob
import numpy as np
import os
import pickle as pkl
import subprocess
import unittest

from subprocess import call


class RunTests(unittest.TestCase):
    """
    RunTests inherits from unittest.TestCase.

    Pickle data is loaded for testing dimensions.

    API generate_numbers_sequence() is called for testing properties of the
    returned array.

    Script 'python generate_numbers_sequence.py' is run for testing properties
    of the saved image sequence.

    Existing sequence filenames are stored in a list for testing properties of
    the filenames.
    """

    data = pkl.load(open('mnist.pkl', 'rb'), encoding='latin1')

    height = int(np.sqrt(data[0][0][0].shape[0]))
    width = 100

    sequence = generate_numbers_sequence.generate_numbers_sequence([1], (4, 5),
                                                                   width)

    output = call(['python', 'generate_numbers_sequence.py', '-d', '1', '-r',
                   '4', '5', '-w', '100', '-a', 'mnistm'],
                  stdout=subprocess.PIPE)

    filenames = glob.glob1(os.getcwd(), "sequence*")

    def test_MnistPklFile(self):
        """Test pickle data dimensions."""
        self.assertEqual(RunTests.data[0][0].shape, (50000, 784),
                         'Training data shape is not (50000, 784)')

        self.assertEqual(RunTests.data[1][0].shape, (10000, 784),
                         'Test data shape is not (10000, 784)')

        self.assertEqual(RunTests.data[2][0].shape, (10000, 784),
                         'Validation data shape is not (10000, 784)')

        self.assertEqual(RunTests.data[0][1].shape, (50000, ),
                         'Training label shape is not (50000, )')

        self.assertEqual(RunTests.data[1][1].shape, (10000, ),
                         'Testing label shape is not (10000, )')

        self.assertEqual(RunTests.data[2][1].shape, (10000, ),
                         'Validation label shape is not (10000, )')

    def test_StackDims(self):
        """Test stacked data dimensions."""
        X = np.vstack((RunTests.data[0][0],
                       RunTests.data[1][0],
                       RunTests.data[2][0]))

        y = np.hstack((RunTests.data[0][1],
                       RunTests.data[1][1],
                       RunTests.data[2][1]))

        self.assertEqual(X.shape, (70000, 784),
                         'Stacked X-shape is not equivalent to stacked mnist \
                          data.')

        self.assertEqual(y.shape, (70000,),
                         'Stacked y-shape is not equivalent to stacked mnist \
                         labels.')

    def test_ImageHeight(self):
        """Test generated sequence height."""
        self.assertEqual(RunTests.sequence[0].shape[0], RunTests.height,
                         'Sequence height is not equal to 28.')

    def test_ImageWidth(self):
        """Test generated sequence width."""
        self.assertEqual(RunTests.sequence[0].shape[1], RunTests.width,
                         'Sequence width is not equal to input width.')

    def test_ImageType(self):
        """Test generated sequence dtype."""
        self.assertEqual(RunTests.sequence[0].dtype, np.dtype('float32'),
                         'Sequence is not a float type.')

    def test_ImageElementRange(self):
        """Test generated sequence element range."""
        self.assertEqual((RunTests.sequence[0] > 1).sum(), 0,
                         'Sequence contains values greater than 1.')

    def test_ScriptRuntimeSuccess(self):
        """Test for script runtime errors."""
        self.assertEqual(RunTests.output, 0,
                         'Script exits with a non 0 error code.')

    def test_FileExts(self):
        """Test all generated sequences for a .png extension."""
        for filename in RunTests.filenames:
            self.assertTrue('.png' in filename,
                            'Saved sequence is not a .png file')


if __name__ == '__main__':
    unittest.main()
