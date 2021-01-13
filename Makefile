train_model:
	python3 run.py train \
		--train '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/speaker1 data/vctk-speaker1-train.8.16000.-1.4096.h5' \
		--val '/home/taimur/Documents/Online Courses/Fourth Brain/Projects/Audio_super_res/speaker1 data/vctk-speaker1-val.8.16000.6.4096.h5' \
		--epochs 120 \
		--batch-size 64 \
		--r 4 \
		--layers 4 \
		--pool_size 8 \
		--strides 8

