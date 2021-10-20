#!/bin/bash

# -ss start at 680 seconds (11m 30s)
# -sseof end at 12 minutes
# -vsync 0 all frames
ffmpeg -ss 680 \
       -sseof 720 \
       -threads 4 \
       -i data/ProgDieFeed_20211014000000.avi \
       -vsync 0 \
       -frame_pts true \
       -f image2 'data/frames/img-%d.png'


