#!/bin/sh

set -euxo pipefail

palette="/tmp/palette.png"
filters="fps=60,scale=1024:-1:flags=lanczos"

if [ -f $palette ]; then rm $palette; fi

ffmpeg -v warning -i $1 -vf "$filters,palettegen" -y $palette
ffmpeg -v warning -stats -i $1 -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y $2
