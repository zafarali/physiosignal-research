#!/bin/bash
muse-io --preset 14 --device Muse-2B85 --osc osc.udp://localhost:5000 &
muse-player -l 5000 -s osc.udp://localhost:5001 &
python proto-script.py