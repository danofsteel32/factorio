#!/bin/bash

mkdir -p data/frames

#       -frame_pts true \

# Positives (11m - 11m39s)
ffmpeg -ss 00:10:00 \
       -to 00:12:00 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 2 \
       -q:v 2 \
       -f image2 'data/frames/%d.jpg'
