#! /usr/bin/env python3

import unittest
from performance_statistics import PerformanceStatistics
import os
import sys

class TestPerformanceStatistics(unittest.TestCase):
    def testSimpleInputFile(self):
        test_file = os.path.dirname(sys.argv[0]) + '/performance_statistics_test_data.csv'
        test_benchmark = 'my_gups/1GB'
        ps = PerformanceStatistics(test_file, 'benchmark')
        self.assertListEqual(ps.getIndexColumn().tolist(), ['my_gups/16GB', 'my_gups/1GB', 'my_gups/4GB'])
        self.assertEqual(ps.getWalkDuration(test_benchmark), 42850778964.0)
        self.assertEqual(ps.getStlbMisses(test_benchmark), 554443700.0)
        self.assertEqual(ps.getRuntime(test_benchmark), 31148327321.0)

unittest.main()

