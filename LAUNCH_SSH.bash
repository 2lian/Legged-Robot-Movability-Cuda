#!/bin/bash

 # ssh moonbotj40@10.240.20.144 -yes
 # cd elian_stuff/
 # rm -rf cuda_research_ssh_tmp/
 # mkdir cuda_research_ssh_tmp
 # exit
SSH_ADDRESS="moonbotj40@10.240.20.144"
SOURCE_DIR="./"
DESTINATION_DIR="${SSH_ADDRESS}:~/elian_stuff/cuda_research_ssh_tmp/"

rsync -av --exclude='externals/cuda-12.2_linux' --exclude='*Cache*' --exclude='*_win' $SOURCE_DIR $DESTINATION_DIR

# find "$SOURCE_DIR" -type f -not -path '*/\.*' ! $EXCLUDE_DIRS -exec scp {} "$DESTINATION_DIR" \;
ssh "${SSH_ADDRESS}" <<EOF
script -q /dev/null
cd ~/elian_stuff/cuda_research_ssh_tmp/
. LAUNCH.bash
EOF
# Run your commands here
# scp -r . moonbotj40@10.240.20.144:/elian_stuff/cuda_research_ssh_tmp/. 
