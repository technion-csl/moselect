#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from layouts_generator import LayoutsGenerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-n', '--num_layouts', type=int, default=9)
parser.add_argument('--use_1gb_hugepages', action='store_true', default=False)
parser.add_argument('--max_1gb_hugepages', type=int, default=4)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

generator = LayoutsGenerator(args.memory_footprint, args.num_layouts, args.use_1gb_hugepages)
generator.buildGrowingWindowLayouts(args.max_1gb_hugepages)
generator.exportLayouts(args.output)


