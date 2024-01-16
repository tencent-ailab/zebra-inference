# Doc for Zebra-Inference
This Repo is release for play with `Zebra-7b-v2-lit` model.
This Repo is forked from [Huggingface Transformers-Bloom-Inference](https://github.com/huggingface/transformers-bloom-inference).
We Implement the zebra model as a patch to original transformers and directly utilize the mentioned repo as an interface to our model.

## Resource

- Paper: [Zebra: Extending Context Window with Layerwise Grouped Local-Global Attention](https://arxiv.org/pdf/2312.08618.pdf)
- Model: [Huggingface Model Repo](https://huggingface.co/kqsong/zebra-7b-lcat-v2-lit)

## Citation
```
@article{song2023zebra,
      title={Zebra: Extending Context Window with Layerwise Grouped Local-Global Attention}, 
      author={Kaiqiang Song and Xiaoyang Wang and Sangwoo Cho and Xiaoman Pan and Dong Yu},
      year={2023}
}
```

## Play with Zebra
### Step 1: Enviorment
Build a conda enviorment throught the below command line.
```shell
conda create -n zebra-inference python=3.9
conda activate zebra-inference
conda install -c anaconda cmake -y

pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
    transformers==4.31.0 \
    deepspeed==0.7.6 \
    accelerate==0.20.3 \
    gunicorn==20.1.0 \
    flask==2.3.0 \
    werzeug==2.3.0 \
    flask_api \
    fastapi==0.89.1 \
    uvicorn==0.19.0 \
    jinja2==3.1.2 \
    pydantic==1.10.2 \
    grpcio-tools==1.50.0 \
    sentencepiece \
    --no-cache-dir
```

### Step 2: Download the model from HF-models
Download the model with git.
```shell
git lfs install
git clone https://huggingface.co/kqsong/zebra-7b-lcat-v2-lit
```

### Step 3: Launch the Zebra
```shell
bash launch_zebra_all.sh <path/to/model>
```
This will launch both frontend(port:5001) and backend(port:5000).

### Step 4: Open Web Browser to play with Zebra
Please visit the localhost with port, or through your ip address and the port.
```
https://0.0.0.0:5051
https://<ip_address>:5051
```

## LICENSE
- The code licensed under the [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0)
- The model licensed under the [Llama-2 License](https://ai.meta.com/llama/license/)

## Disclaimer
This repo is only for research purpose. It is not an officially supported Tencent product.