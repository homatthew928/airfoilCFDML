#!/bin/sh

set -e

cd "${0%/*}" || exit 1

templateDir="template"

projectDir="/path/to/airfoilCFDML"

validationDir="${projectDir}/gmsh/validationMesh4"

echo "========================================"
echo "NACA 0012 validation sweep"
echo "========================================"

for casePath in "$validationDir"/*
do
    caseName=$(basename "$casePath")
    case "$caseName" in
        n*)
            AoA="-${caseName#n}"
            ;;
        *)
            AoA="$caseName"
            ;;
    esac

    echo "========================================"
    echo "Preparing AoA = ${AoA} degrees"
    echo "Case directory = ${caseName}"
    echo "========================================"

    cp -r "$templateDir" "$caseName"

    "$caseName/Allrun" "$AoA"

    echo "========================================"
    echo "Completed AoA = ${AoA} degrees"
    echo "========================================"

done

echo
echo "========================================"
echo "All validation cases completed"
echo "========================================"
