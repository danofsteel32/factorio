#!/bin/bash


ffmpeg -start_number 1000 -i data/overlay_frames/%d.jpg -vcodec h264 data/test.avi
