#!/bin/bash
#SBATCH -A SNIC2020-9-180
#SBATCH --time=02:00:00
#SBATCH -c 14
##SBATCH -p largemem
##SBATCH --mem 768G
#SBATCH -J janrei_aqua

echo "Which folder/file would you like to analyze?"
read -p "Input: " input
echo "Analyzing: $input"

read -p "Preset: " preset
read -p "First frame: " firstframe
read -p "#channels: " totalchannels

# load libraries
ml MATLAB/2019b.Update2

# go to folder
cd ~/Private/aqua_janrei/

# run script
#srun matlab -r "preset=$preset;file='$file';try, run('aqua_cmd_dynamic.m'); end; quit"
#srun matlab -r "try, run('aqua_cmd.m'); end; quit"
srun matlab -r "preset='total_channels='$totalchannels';$preset';channel='$firstframe';input='$input';run('aqua_cmd.m');"
