#!/bin/sh


for file in ./cluster_scripts/*
do
  sbatch "$file" 
done













