#!/bin/bash
for omega in {60..100..5}
do
    python enhancement.py -o $omega -c
done
for t0 in {0..40..5}
do
    python enhancement.py -t $t0 -c
done
for dist in {1..8}
do
    python enhancement.py -d $dist -c
done
