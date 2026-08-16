python dub_pipeline_v2.py \
	--input $1 \
	--output $2 \
	--workdir $3 \
	--device cpu \
	--hf-token $HF_TOKEN \
	--translator sarvam \
	--tts-method chatterbox \
	--no-use-subs
