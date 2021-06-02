export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=3

nohup python -u driver/Train.py --config ddp.cfg.S > log 2>&1 &
tail -f log
