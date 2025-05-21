#!/bin/bash

for file in "$1"/*
do
    echo "$file"
    python main-train.py "$file"
done