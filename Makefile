train_model:
	python3 run.py \
		--train ../speaker1_data/vctk-speaker1-train.8.16000.-1.4096.h5 \
		--val ../speaker1_data/vctk-speaker1-train.8.16000.6.4096.h5 \
		--e 120 \
		--batch-size 64 \
		--lr 1e-3 \
		--r 4 \
		--layers 4 \
		--piano false \
		--pool_size 8 \
		--strides 8 \
		--full true

