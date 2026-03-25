#!/usr/bin/env bash

# rename gpus
source /nethome/rhakim/projects/deepconfTesting/scripts/rename_gpus.sh

# setup basic paths
export PROJECT_DIR=/nethome/rhakim/projects/deepconfTesting
#export OUTPUT_DIR=/data/users/rhakim/logs/deepconfTesting
export CACHE_BASE_DIR=/data/users/zakhalili/models

# cd to project dir
cd $PROJECT_DIR
