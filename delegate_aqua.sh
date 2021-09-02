echo "Distributing batch tasks for : $1 ..."

sbatch run_aqua.bash -i $1 -s 1 -e 2000
sbatch run_aqua.bash -i $1 -s 2000 -e 4000
sbatch run_aqua.bash -i $1 -s 4000 -e 6000
sbatch run_aqua.bash -i $1 -s 6000 -e 8000
sbatch run_aqua.bash -i $1 -s 8000 -e 10000

echo "Distributed!"
