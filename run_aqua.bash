#!/bin/bash
#SBATCH -A SNIC2020-9-180
#SBATCH --time=08:00:00
#SBATCH -c 28
##SBATCH -p largemem
##SBATCH --mem 300G
#SBATCH -J janrei_aqua

while getopts i:s:e: flag
do
	case "${flag}" in
		s) start=${OPTARG};;
		e) end=${OPTARG};;
		i) file=${OPTARG};;
	esac
done

# check if flags are set
if [ -z ${file+x} ]; then echo "Please set a file!"; exit 1; else echo "Processing of file $file"; fi 

# load libraries
ml MATLAB/2019b.Update2

# go to folder
cd ~/Private/aqua_janrei2/

# run script
#srun matlab -r "preset=$preset;file='$file';try, run('aqua_cmd_dynamic.m'); end; quit"
#srun matlab -r "try, run('aqua_cmd.m'); end; quit"
#srun matlab -r "preset=$preset;file='$file';run('keb_cmd.m');"

if [ -z ${start+x} ]; then srun matlab -r "indices=[$start $end];file='$file';run('keb_cmd.m');"; else srun matlab -r "file='$file';run('keb_cmd.m');"; fi
