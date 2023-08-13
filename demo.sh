#!/bin/bash

# CPMR
python main.py --cuda 0 --data garden --lr 2e-2 --l2 2e-2 --len_ctx 5

python main.py --cuda 1 --data video --lr 5e-3 --l2 5e-3 --len_ctx 35

python main.py --cuda 0 --data game --no_bn --lr 2e-3 --l2 2e-4 --len_ctx 50

python main.py --cuda 0 --data ml --lr 2e-3 --l2 2e-3 --len_ctx 35 --lr_step 6
