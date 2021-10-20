#!/usr/bin/env python

import argparse
from pathlib import Path
from imv_wrapper import ImvWindow

parser = argparse.ArgumentParser(description='Do labeling shit')
parser.add_argument('-p', '--pos', action='store_true', help='positives')
parser.add_argument('-n', '--neg', action='store_true', help='negatives')
args = parser.parse_args()

if not (args.pos or args.neg):
    print('-p or -n')
    exit()

dir_ = 'pos' if args.pos else 'neg'
images = sorted([
    i for i in (Path('data') / dir_).iterdir()
], key=lambda x: int(x.name.split('-')[-1].split('.')[0]))
ImvWindow.view_images(images, dir_)
