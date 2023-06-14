#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=167:50:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=18G
#SBATCH --mail-type=END


module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate skimpy-env


cd /tudelft.net/staff-bulk/ewi/insy/DBL/plent/Benchmark/Benchmark/scripts



srun --ntasks=1 python3 100123_cs_pathway_I.py 7500000 7550000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7550000 7600000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7600000 7650000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7650000 7700000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7700000 7750000 &

srun --ntasks=1 python3 100123_cs_pathway_I.py 7750000 7800000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7800000 7850000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7850000 7900000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7900000 7950000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 7950000 8000000 &


srun --ntasks=1 python3 100123_cs_pathway_I.py 9100000 9150000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9150000 9200000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9200000 9250000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9250000 9300000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9300000 9350000 &

srun --ntasks=1 python3 100123_cs_pathway_I.py 9350000 9400000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9400000 9450000 &
srun --ntasks=1 python3 100123_cs_pathway_I.py 9450000 9500000 &
wait











