#!/bin/sh

python -m domainbed.scripts.sweep supervised_adaptation\
       --data_dir=./dataset \
       --output_dir=./sweep/$1 \
       --command_launcher $5\
       --algorithms ERM\
       --datasets $2\
       --n_hparams 1\
       --n_trials_from $3 \
       --n_trials $4 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation

