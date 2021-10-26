#!/bin/sh

# <Argments>
# $1 : backbone
# $2 : datasets
# $3 : algorithms (ERM, CORAL, etc)
# 
# <Example>
# sh scripts/hparam_search.sh resnet50 PACS ERM
# sh scripts/hparam_search.sh resnet50 PACS CORAL  # different base algorithm


python -m domainbed.scripts.sweep delete_incomplete\
       --data_dir=./dataset \
       --output_dir=./sweep_hparam/$1 \
       --command_launcher multi_gpu\
       --algorithms $3\
       --datasets $2\
       --n_hparams_from 0 \
       --n_hparams 20 \
       --n_trials_from 0 \
       --n_trials 3 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation

python -m domainbed.scripts.sweep launch\
       --data_dir=./dataset \
       --output_dir=./sweep_hparam/$1 \
       --command_launcher multi_gpu\
       --algorithms $3\
       --datasets $2\
       --n_hparams_from 0 \
       --n_hparams 20\
       --n_trials_from 0 \
       --n_trials 3 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation

python -m domainbed.scripts.sweep unsupervised_adaptation\
       --data_dir=./dataset \
       --output_dir=./sweep_hparam/$1 \
       --command_launcher multi_gpu\
       --algorithms $3\
       --datasets $2\
       --n_hparams_from 0 \
       --n_hparams 20\
       --n_trials_from 0 \
       --n_trials 3 \
     --single_test_envs \
     --hparams "{\"backbone\": \"$1\"}" \
     --skip_confirmation