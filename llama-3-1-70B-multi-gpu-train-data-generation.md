nohup /opt/bin/cuda-reserve.py --num-gpus 6 accelerate launch --config_file=multi_gpu_acc.yaml  ge_data_all_llama3-1-70Bchat.py --start=0 --end=68000 --index=0 --gpu_index 0 --outdir ../../train-data-llama-3-1-70b/sharegpt_0_67999_mufp16 > nohup_data_gen_script.out &

refer `eagle/ge_data/multi_gpu_acc.yaml` training accelerate config

I modified ge_data_all-llama3-1chat-70b.py