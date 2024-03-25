#!/bin/bash
set -e -o pipefail
 # ssh moonbotj40@10.240.20.224 -yes
 # cd elian_stuff/
 # rm -rf cuda_research_ssh_tmp/
 # mkdir cuda_research_ssh_tmp
 # exit
SSH_ADDRESS=${address_jetson}
SOURCE_DIR="./"
DESTINATION_DIR="${SSH_ADDRESS}:~/elian_cuda_tmp/"

rsync -av --exclude-from='.gitignore' --exclude='*.bin' --exclude='image' --exclude='externals/cuda-12.2_linux' --exclude='*Cache*' --exclude='*_win' $SOURCE_DIR $DESTINATION_DIR

# find "$SOURCE_DIR" -type f -not -path '*/\.*' ! $EXCLUDE_DIRS -exec scp {} "$DESTINATION_DIR" \;
ssh -q "${SSH_ADDRESS}" <<EOF
script -q /dev/null
cd ~/elian_cuda_tmp
. LAUNCH.bash
exit
EOF
# Run your commands here
# scp -r . moonbotj40@10.240.20.224:/elian_stuff/cuda_research_ssh_tmp/.
#
rsync -av --include='*.txt' --include='*.png' --exclude='*' "${DESTINATION_DIR}" "${SOURCE_DIR}/image"
rsync -av --include='*.npy' --exclude='*' "${DESTINATION_DIR}" "/home/elian/moonbot_software/src/pcl_reader/pcl_reader/python_package_include/"

# rsync -av --include='*.bin' --exclude='*' "${DESTINATION_DIR}" "${SOURCE_DIR}"
# python3 array_vizu_1D.py
