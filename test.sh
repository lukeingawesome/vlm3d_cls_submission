#!/usr/bin/env bash
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" && pwd )"

./build.sh


VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

docker volume create ctlipro-output-$VOLUME_SUFFIX


docker run --rm \
        --gpus all \
        --memory="$MEM_LIMIT" \
        --memory-swap="$MEM_LIMIT" \
        --network bridge \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --shm-size 128m \
        --pids-limit 256 \
        -v "$SCRIPTPATH/test/":/input/ \
        -v ctlipro-output-$VOLUME_SUFFIX:/output/ \
        ctlipro


# Save results to host for debugging
mkdir -p "$SCRIPTPATH/debug_output"
docker run --rm \
        -v ctlipro-output-$VOLUME_SUFFIX:/output/ \
        -v "$SCRIPTPATH/debug_output":/debug/ \
        nvidia/cuda:12.2.0-devel-ubuntu22.04 cp /output/results.json /debug/

echo "=== First 10 lines of results.json ==="
head -10 "$SCRIPTPATH/debug_output/results.json"
echo "=== Last 10 lines of results.json ==="
tail -10 "$SCRIPTPATH/debug_output/results.json"

echo "=== Attempting JSON validation ==="
python3 -m json.tool "$SCRIPTPATH/debug_output/results.json" > /dev/null && echo "JSON is valid" || echo "JSON is invalid"

docker volume rm ctlipro-output-$VOLUME_SUFFIX
