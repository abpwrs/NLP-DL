#!/bin/sh
# Author: AlexPowers
# Mon Nov  5 19:37:12 CST 2018
# This script must be run from project root
if [ ${#} -ne 1 ]; then
    echo "Incorrect Usage!"
    echo "${0} PROJECT_NAME"
    exit
fi

for dir in data model out src job; do
    rm -rf ${dir}/$1
done