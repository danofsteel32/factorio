#!/bin/bash

mkdir -p data/{pos,neg}

# -t total duration in seconds
# -vsync 0 all frames

# Positives (11m - 11m39s)
ffmpeg -ss 00:10:00 \
       -to 00:11:39 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 0 \
       -frame_pts true \
       -q:v 2 \
       -f image2 'data/pos/%d.jpg'

# Negatives (11m40s - 11m55s)
ffmpeg -ss 00:11:40 \
       -to 00:11:50 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 0 \
       -frame_pts true \
       -q:v 2 \
       -f image2 'data/neg/%d.jpg'
