#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

DATA_DIR='../data/'
mkdir -p $DATA_DIR

cd $DATA_DIR
wget https://storage.googleapis.com/ravens-assets/block-insertion.zip
unzip -q block-insertion.zip
