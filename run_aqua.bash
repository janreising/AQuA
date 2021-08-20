#!/bin/bash
#SBATCH -A SNIC2020-9-180
#SBATCH --time=02:00:00
#SBATCH -c 28
##SBATCH -p largemem
##SBATCH --mem 768G
#SBATCH -J janrei_aqua

echo "Analyzing: $1"
echo "Preset: $2"
echo "First frame: $3"
echo "Scale: $4"

# load libraries
ml MATLAB/2019b.Update2

# go to folder
cd ~/Private/aqua_janrei/

# run script
#srun matlab -r "preset=$preset;file='$file';try, run('aqua_cmd_dynamic.m'); end; quit"
#srun matlab -r "try, run('aqua_cmd.m'); end; quit"
srun matlab -r "scale=$4;preset=$2;input='$1';channel=$3;run('aqua_cmd.m');"

