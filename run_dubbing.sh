python dub_pipeline_v2.py \
	--input $1 \
	--output $2 \
	--workdir $3 \
	--device mps \
	--hf-token $HF_TOKEN \
	--translator sarvam \
	--clap-model \
	--tts-method parler \
	--no-use-subs
