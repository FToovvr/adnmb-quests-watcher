#!/bin/sh

set -e

for i in $(seq 1 3); do
    cat 1_create_tables_${i}_*.psql
    echo
done

for i in $(seq 1 2); do
    cat 2_create_functions_${i}_*.psql
    echo
done

for i in $(seq 1 2); do
    cat 3_create_procedures_${i}_*.psql
    echo
done