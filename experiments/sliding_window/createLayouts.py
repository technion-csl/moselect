#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(sys.argv[0])+"/..")
from layouts_generator import LayoutsGenerator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
parser.add_argument('-r', '--hot_region', default='hot_region.txt')
parser.add_argument('-n', '--num_layouts', type=int, default=9)
parser.add_argument('-t', '--weight', type=int, default=50)
parser.add_argument('-g', '--use_1gb_pages', action='store_true')
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

import pandas as pd
hot_region_df = pd.read_csv(args.hot_region, index_col=False, comment='#')
conf_row = hot_region_df[hot_region_df['window-weight'] == args.weight]
hot_region_start_page = conf_row['window-start'].iloc[0]
hot_region_num_pages = conf_row['window-length'].iloc[0]

generator = LayoutsGenerator(args.memory_footprint, args.num_layouts, args.use_1gb_pages)
if hot_region_start_page < 0 or hot_region_num_pages < 0:
    print('The benchmark has no ' + str(args.weight) + '% weighted window')
    print('building random-window layouts instead with seed=' + str(args.weight))
    generator.buildRandomWindowLayouts(seed=args.weight)
else:
    hot_region_start =  hot_region_start_page * 4096
    hot_region_length = hot_region_num_pages * 4096
    generator.buildSlidingWindowLayouts(hot_region_start, hot_region_length)
generator.exportLayouts(args.output)

