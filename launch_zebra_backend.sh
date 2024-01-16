MODEL_PATH=$1

TOKENIZERS_PARALLELISM=false \
MODEL_NAME=${MODEL_PATH} \
MODEL_CLASS=ZebraModelForCausalLM \
DEPLOYMENT_FRAMEWORK=hf_accelerate \
DTYPE=bf16 \
MAX_INPUT_LENGTH=16384 \
ALLOWED_MAX_NEW_TOKENS=1024 \
MAX_BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DEBUG=false \
gunicorn -t 0 -w 1 -b 0.0.0.0:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s' &

