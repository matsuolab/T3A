#!/bin/sh
nvidia-smi


python -m domainbed.scripts.sweep unsupervised_adaptation\
       --data_dir=./dataset \
       --output_dir=./sweep/$1 \
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets $2\
       --n_hparams 1\
       --n_trials_from $3 \
       --n_trials $4 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation
