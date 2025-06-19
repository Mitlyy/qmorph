#!/bin/bash

URL="https://drive.google.com/file/d/1byGTGLx1jJz-VX7R6jZZyexBzJFsgrRm/view?usp=drive_link"

ZIPFILE="checkpoints.zip"
wget "$URL" -O "$ZIPFILE"
unzip "$ZIPFILE" -d .
rm "$ZIPFILE"
