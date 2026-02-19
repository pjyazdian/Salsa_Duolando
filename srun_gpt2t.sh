#!/bin/sh
currenttime=`date "+%Y%m%d%H%M%S"`
if [ ! -d log ]; then
    mkdir log
fi

echo "[Usage] ./srun_gpt2t.sh config_path [train|eval|demo|visgt|anl|sample] partition gpunum"
# check config exists
if [ ! -e "$1" ]; then
    echo "[ERROR] configuration file: $1 does not exist!"
    exit 1
fi

# expname/config_suffix: derive from config path for mkdir and job name (fixes empty variables)
config_basename=$(basename "$1" .yaml)
config_suffix="gpt_${config_basename}"
expname="experiments"
if [ ! -d "$expname" ]; then
    mkdir "$expname"
fi
echo "[INFO] saving results to, or loading files from: $expname"

if [ "$3" == "" ]; then
    echo "[ERROR] enter partition name"
    exit
fi
partition_name=$3
echo "[INFO] partition name: $partition_name"

if [ "$4" == "" ]; then
    echo "[ERROR] enter gpu num"
    exit
fi
gpunum=$4
gpunum=$(($gpunum<8?$gpunum:8))
echo "[INFO] GPU num: $gpunum"
((ntask=$gpunum*3))


# Solar: specify --mem and --time (guideline: always specify memory). Add --account if required (e.g. --account=3dlg-hcvc-lab).
TOOLS="srun --partition=$partition_name --gres=gpu:$gpunum --cpus-per-task=32 --ntasks-per-node=1 -n1 --job-name=$config_suffix --mem=128G --time=3-00:00"
PYTHONCMD="python -u main_gpt2t.py --config $1"

if [ $2 == "train" ];
then
    $TOOLS $PYTHONCMD \
    --train 
elif [ $2 == "eval" ];
then
    $TOOLS $PYTHONCMD \
    --eval 
elif [ $2 == "demo" ];
then
    $TOOLS $PYTHONCMD \
    --demo 
elif [ $2 == "visgt" ];
then
    $TOOLS $PYTHONCMD \
    --visgt 
elif [ $2 == "anl" ];
then
    $TOOLS $PYTHONCMD \
    --anl 
elif [ $2 == "sample" ];
then
    $TOOLS $PYTHONCMD \
    --sample 
fi

