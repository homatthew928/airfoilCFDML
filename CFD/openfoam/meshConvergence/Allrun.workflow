#!/bin/sh

set -e

cd "${0%/*}" || exit 1

templateDir="template"

projectDir="/path/to/airfoilCFDML"

meshConvergenceDir="${projectDir}/gmsh/meshConvergence"

echo "========================================"
echo "NACA 0012 mesh convergence study"
echo "AoA 10.12"
echo "========================================"

for casePath in "$meshConvergenceDir"/*
do
    caseName=$(basename "$casePath")

    echo "========================================"
    echo "Preparing mesh level ${caseName#mesh}"
    echo "========================================"

    cp -r "$templateDir" "$caseName"

    "$caseName/Allrun" "10.12" "$caseName"

    echo "========================================"
    echo "Completed mesh level ${caseName#mesh}"
    echo "========================================"

done

echo
echo "========================================"
echo "All mesh convergence cases completed"
echo "========================================"
