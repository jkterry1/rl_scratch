CUDA_VISIBLE_DEVICES=0 nohup python3 run_rllib.py &> 0.out &
CUDA_VISIBLE_DEVICES=0 nohup python3 run_rllib.py &> 1.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 run_rllib.py &> 2.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 run_rllib.py &> 3.out &
CUDA_VISIBLE_DEVICES=2 nohup python3 run_rllib.py &> 4.out &
CUDA_VISIBLE_DEVICES=2 nohup python3 run_rllib.py &> 5.out &
CUDA_VISIBLE_DEVICES=3 nohup python3 run_rllib.py &> 6.out &
CUDA_VISIBLE_DEVICES=3 nohup python3 run_rllib.py &> 7.out &