#!/usr/bin/env bash
#
# Run a specific test from the examples package.
# Usage: ./scripts/run.sh <test-number>
# Example: ./scripts/run.sh 2  →  runs Test2

set -e

# Verify argument
if [ -z "$1" ]; then
    echo "Error: missing test number."
    echo "Usage: $0 <test-number>"
    echo "Example: $0 2"
    exit 1
fi

TEST_NUM="$1"
TEST_CLASS="com.pierluigicovone.microgradj.examples.Test${TEST_NUM}"

# Move to project root (the script can be invoked from anywhere)
cd "$(dirname "$0")/.."

# Verify the source file exists
TEST_FILE="src/main/java/com/pierluigicovone/microgradj/examples/Test${TEST_NUM}.java"
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: test file not found: $TEST_FILE"
    echo "Available tests:"
    ls src/main/java/com/pierluigicovone/microgradj/examples/ 2>/dev/null | grep -E '^Test[0-9]+\.java$' | sed 's/^/  /'
    exit 1
fi

# Suppress Maven JVM warnings on Java 24+
export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED --sun-misc-unsafe-memory-access=allow"

echo "Running $TEST_CLASS..."
echo "----------------------------------------"

mvn -q compile exec:java \
    -Dexec.mainClass="$TEST_CLASS" \
    -Dexec.cleanupDaemonThreads=false