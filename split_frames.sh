#!/bin/bash

mkdir -p data/{pos,neg}

# -t total duration in seconds
# -vsync 0 all frames

# Positives (11m - 11m39s)
ffmpeg -ss 660 \
       -t 39 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 0 \
       -frame_pts true \
       -f image2 'data/pos/pos-%d.png'

# Negatives (11m40s - 11m55s)
ffmpeg -ss 700 \
       -t 15 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 0 \
       -frame_pts true \
       -f image2 'data/neg/neg-%d.png'
