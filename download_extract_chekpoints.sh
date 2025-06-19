#!/bin/bash

URL="https://drive.google.com/drive/folders/18vWPf63O8CPGyS8NZQvAlYMSz6N7ewlf?usp=sharing/"

ZIPFILE="checkpoints.zip"
wget "$URL" -O "$ZIPFILE"
unzip "$ZIPFILE" -d .
rm "$ZIPFILE"
