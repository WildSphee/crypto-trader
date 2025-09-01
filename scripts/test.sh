#!/usr/bin/env bash

if [ $# -eq 0 ];
then
  echo "Defaulting to apply on '.'"
  FPATH="."
else
 FPATH=$1
fi

export $(grep -v '^#' .env | xargs)

# to run this `poetry add --dev pytest-cov coverage`
pytest \
    --cov-config scripts/.coveragerc \
    --cov-report term-missing \
    --cov --cov-report html \
    --cov-context test \
    --cov-fail-under 70 \
    -v -r fEsX $FPATH
