#!/bin.sh $6

# <Arguments>
# $1 : mode (pretrain/sup/unsup)
# $2 : backbone name (resnet50, ViT-B16, etc)
# $3 : n_traials_from 
# $4 : n_trials 
# $5 : command launcher (local/multi_gpu) 
# 
# <Example>
# sh scripts/launch.sh pretrain resnet50 10 3 local
# sh scripts/launch.sh sup resnet50 10 3 local
# sh scripts/launch.sh unsup resnet50 10 3 local


if [ $1 = "unsup" ]; then
    echo "unsupervised adaptation"
    sh scripts/unsup.sh $2 PACS $3 $4 $5
    sh scripts/unsup.sh $2 VLCS $3 $4 $5
    sh scripts/unsup.sh $2 OfficeHome $3 $4 $5
    sh scripts/unsup.sh $2 TerraIncognita $3 $4 $5
elif [ $1 = "sup" ]; then
    echo "supervised adaptation"
    sh scripts/sup.sh $2 PACS $3 $4 $5
    sh scripts/sup.sh $2 VLCS $3 $4 $5
    sh scripts/sup.sh $2 OfficeHome $3 $4 $5
    sh scripts/sup.sh $2 TerraIncognita $3 $4 $5
elif [ $1 = "pretrain" ]; then
    echo "pretraining the ERM model"
    sh scripts/source.sh $2 PACS $3 $4 $5
    sh scripts/source.sh $2 VLCS $3 $4 $5
    sh scripts/source.sh $2 OfficeHome $3 $4 $5
    sh scripts/source.sh $2 TerraIncognita $3 $4 $5
else
    echo "Invalid option"
fi
