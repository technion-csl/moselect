#! /usr/bin/env python3

import sys
import pandas as pd
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', type=argparse.FileType('r'),
                    default=sys.stdin,
                    help='input file with access summary file to plot')
parser.add_argument('-o', '--output_file', required=True,
                    help='the output file to save the plot in')
parser.add_argument('-t', '--time_windows', type=int, default=1,
                    help='the number of time windows to split the trace to')
parser.add_argument('-n', '--normalize', action='store_true',
                    help='specify if to normalize data before plot')
parser.add_argument('--figure_y_label', required=False,
                    default='memory accesses',
                    help='the label of plotted figure y-axis')
parser.add_argument('-p', '--pool', choices=['mmap', 'brk'], default='brk',
                    help='the pool to plot its accesses')
args = parser.parse_args()

df = pd.read_csv(args.input_file, delimiter=',')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

df = df[df['PAGE_TYPE'].str.contains(args.pool)]
df = df[['PAGE_NUMBER', 'NUM_ACCESSES']]
# normalize and split data to plot
num_rows = df['PAGE_NUMBER'].size
window_rows = math.ceil(num_rows / args.time_windows)
access_sum = df['NUM_ACCESSES'].sum()
output_file = args.output_file
#plot bars
with PdfPages(output_file) as pdf:
    for w in range(args.time_windows):
        start_row = w*window_rows
        end_row = (w+1)*window_rows - 1
        # split and copy
        df_to_plot = df[start_row:end_row].copy()
        num_pages = df_to_plot['PAGE_NUMBER'].max()
        # group by page number
        #df_to_plot = df_to_plot.groupby(['PAGE_NUMBER'], sort=False).size().rename('NUM_ACCESSES').reset_index()
        # normalize
        if (args.normalize):
            df_to_plot['NUM_ACCESSES'] /= access_sum
        #plot bins bars
        plt.bar(df_to_plot['PAGE_NUMBER'], df_to_plot['NUM_ACCESSES'])
        plt.gca().yaxis.grid(True)
        #plt.grid(True)
        plt.xlabel('page number')
        plt.ylabel(args.figure_y_label)
        #plt.yscale('log')

        plt.title('all pages')
        plt.bar(df_to_plot['PAGE_NUMBER'], df_to_plot['NUM_ACCESSES'])
        plt.gca().yaxis.grid(True)
        pdf.savefig()
        plt.close()

        plt.title('first 10%')
        plt.bar(df_to_plot['PAGE_NUMBER'], df_to_plot['NUM_ACCESSES'])
        plt.gca().yaxis.grid(True)
        plt.xlim(0, int(num_pages*0.1))
        pdf.savefig()
        plt.close()

        plt.title('last 10%')
        plt.bar(df_to_plot['PAGE_NUMBER'], df_to_plot['NUM_ACCESSES'])

        plt.gca().yaxis.grid(True)
        plt.xlim(int(0.9*num_pages), num_pages)
        pdf.savefig()
        plt.close()
    '''
    #plot accesses bars
    plt.bar(df['NUM_ACCESSES'], df['PAGE_NUMBER'])
    #plt.grid(True)
    plt.gca().yaxis.grid(True)
    plt.ylabel(args.figure_y_label)
    plt.ylabel('total pages')
    #plt.xscale('log')
    pdf.savefig()
    plt.close()
    '''


