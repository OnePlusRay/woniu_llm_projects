CODEFORMER_FIDELITY=0.7
INPUT_PATH=../data/mz/images
proxychains python inference_codeformer.py -w $CODEFORMER_FIDELITY --input_path $INPUT_PATH --bg_upsampler realesrgan