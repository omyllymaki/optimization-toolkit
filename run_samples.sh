#!/usr/bin/env bash
source ./venv/bin/activate
export PYTHONPATH="$PWD"

filenames=`ls samples/*.py`
for file in $filenames
do
  echo "Running file " $file
  python3 $file
done