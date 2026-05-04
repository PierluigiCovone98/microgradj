#!/usr/bin/env bash
#
# Clean compiled artifacts and generated outputs.
# Usage: ./scripts/reset.sh

set -e

# Move to project root
cd "$(dirname "$0")/.."

echo "Cleaning Maven build artifacts..."
export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow"
mvn -q clean

echo "Removing generated graph images..."
rm -f graph.png
rm -f *.png   # remove any png at root
rm -rf out/   # if you decide to use an out/ directory later

echo "Done."