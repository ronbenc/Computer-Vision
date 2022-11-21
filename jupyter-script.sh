#!/bin/bash


unset XDG_RUNTIME_DIR

# xvfb-run -a -s "-screen 0 1440x900x24" 
jupyter notebook --no-browser --ip=$(hostname -I) --port-retries=100