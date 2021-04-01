#!/bin/bash
#SBATCH -A SNIC2020-9-180
#SBATCH --time=02:00:00
#SBATCH -c 14
##SBATCH -p largemem
##SBATCH --mem 768G
#SBATCH -J janrei_aqua

while getopts p:f: flag
do
	case "${flag}" in
		p) preset=${OPTARG};;
		f) file=${OPTARG};;
	esac
done

# check if flags are set
if [ -z ${preset+x} ]; then echo "Please set a preset!"; exit 1; else echo "Processing with preset $preset"; fi 
#if [ -z ${folder+x} ]; then echo "Please set a folder!"; exit 1; else echo "Processing in folder $folder"; fi 
if [ -z ${file+x} ]; then echo "Please set a file!"; exit 1; else echo "Processing of file $file"; fi 


input = $1
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
srun matlab -r "preset='$preset';input='$file';run('aqua_cmd.m');"
