#!/bin/sh

set -e

cd "${0%/*}" || exit 1

templateDir="template"

projectDir="/path/to/airfoilCFDML"

MLDir="${projectDir}/gmsh/MLTest"

lastConfig=0

for casePath in ./*_*
do
    [ -d "$casePath" ] || continue

    caseName=$(basename "$casePath")

    config=${caseName%%_*}

    if [ "$config" -gt "$lastConfig" ]; then
	lastConfig="$config"
    fi
done

firstConfig=$((lastConfig+1))
lastConfigToRun=$((lastConfig+10))

rm -rf [1-9]*

for config in $(seq "$firstConfig" "$lastConfigToRun")
do
    casePath=$(find "$MLDir" -maxdepth 1 -type d -name "${config}_*")

    caseName=$(basename "$casePath")
    AoA="${caseName#*_}"

    echo "========================================"
    echo "Configuration = ${config}"
    echo "Case directory = ${caseName}"
    echo "========================================"

    cp -r "$templateDir" "$caseName"

    "$caseName/Allrun" "$AoA"

    echo "========================================"
    echo "Completed configuration = ${config}"
    echo "========================================"

done

echo
echo "========================================"
echo "Completed configurations ${firstConfig} - ${lastConfigToRun}"
echo "========================================"
