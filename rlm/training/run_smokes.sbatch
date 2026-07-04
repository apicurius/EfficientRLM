#!/bin/bash
# Literal sbatch wrapper for the 3 Qwen3-30B OOLONG reward-style smokes,
# sequential, on one full 8xA6000 node.
#
# NOTE: this can only be SUBMITTED-AND-RUN when you are NOT already at the `ai`
# QoS ceiling. While the holder job (rlm-30b-a6000) is running it pins GPU 8/8
# AND mem 350G/350G for your user, so this sbatch parks forever
# (AssocGrpGRES + QOSMaxMemoryPerUser). To actually run on the held node without
# losing it, use the overlap launcher instead:
#
#   srun --jobid=<HOLDER_JOBID> --overlap --gres=gpu:rtx_a6000:8 -n1 \
#        --cpus-per-task=48 --mem=0 --time=10:00:00 \
#        bash .research/EfficientRLM/rlm/training/run_smokes_seq.sh
#
# Submit this script only on a free 8xA6000 node (or after the holder is
# released):  sbatch run_smokes.sbatch
#
#SBATCH --job-name=rlm-smokes-seq
#SBATCH --partition=ai
#SBATCH --gres=gpu:rtx_a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/omeerdogan23/erlm/.research/EfficientRLM/outputs/rlm-smokes-seq-%j.out
#SBATCH --error=/scratch/omeerdogan23/erlm/.research/EfficientRLM/outputs/rlm-smokes-seq-%j.err

exec bash /scratch/omeerdogan23/erlm/.research/EfficientRLM/rlm/training/run_smokes_seq.sh
