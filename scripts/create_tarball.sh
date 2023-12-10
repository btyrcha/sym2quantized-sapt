#!/usr/bin/env bash
set -xeo

VERSION_TAG="${1:-v0.1.X}"
TEMPORARY_DIR_NAME=sympy-sapt
WORKDIR=$(pwd)
TEMPORARY_LOCATION=$WORKDIR/build/${TEMPORARY_DIR_NAME}

echo "Usage ./scripts/create_tarball.sh <version_tag> run from the root of the repository"
echo "Current workind dir ${WORKDIR}"
echo "Output will be stored in ${TEMPORARY_LOCATION}/.."
mkdir -p $TEMPORARY_LOCATION
rm -rf $TEMPORARY_LOCATION/*

# copy all locations
cp -r examples/ $TEMPORARY_LOCATION/
#cp LICENSE $TEMPORARY_LOCATION/
cd $TEMPORARY_LOCATION/.. &&
    tar -czvf examples_sympy_sapt_${VERSION_TAG}.tar.gz -C $WORKDIR/build $TEMPORARY_DIR_NAME -v &&
    cp examples_sympy_sapt_${VERSION_TAG}.tar.gz $WORKDIR/
