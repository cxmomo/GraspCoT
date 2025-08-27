for ((i=0; i<=7; i++)); do python scripts/create_grasp_data.py --version train --pruning --id $i; done

for ((i=0; i<=1; i++)); do python scripts/create_grasp_data.py --version val --pruning --id $i; done