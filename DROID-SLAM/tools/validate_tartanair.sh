#!/bin/bash


TARTANAIR_PATH=datasets_withaccess/P002

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis $@

