#!/bin/bash

while read -r LINK NAME BASENAME; do
    [ -d "$BASENAME" ] && continue
    URL=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=$LINK" | grep -oP '"href":"\K[^"]+')
    wget --quiet --show-progress -O "$NAME" "$URL"
    mkdir -p "$BASENAME"
    tar -xzf "$NAME" --strip-components=1 -C "$BASENAME"
    rm "$NAME"
done < datasets.txt

