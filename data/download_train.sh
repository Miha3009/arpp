#!/bin/bash

FILENAME="train.tar.gz"
URL=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/qJBuOXVNpWhz0Q" | grep -oP '"href":"\K[^"]+')
wget --quiet --show-progress -O "$FILENAME" "$URL"
mkdir -p "train"
tar -xzf "$FILENAME" --strip-components=1 -C "train"
rm "$FILENAME"
