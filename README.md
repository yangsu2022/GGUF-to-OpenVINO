# GGUF-to-OpenVINO
GGUF Q4 (dequantize) -> PyTorch fp16 -> OpenVINO int4

## Preparation: 
Prepare a server with at least 32GB Mem.

### Install python deps
```bash
pip install -r requirements.txt
```
Refer to [optimum-cli's requirements](https://github.com/openvinotoolkit/openvino.genai/tree/releases/2024/6/samples)
### Download GGUF model
https://huggingface.co/trinhvanhung/Meta-Llama-3.1-8B-Instruct-Q4_K_M/tree/main
```bash
python download_hf_gguf.py
```
Output: `./Meta-Llama-3.1-8B-Instruct-Q4_K_M`
`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` 4.6G

## Step 1: Convert GGUF Model into FP16 PyTorch Model

To convert a GGUF model (such as `Q4_K_M.gguf`) into a PyTorch model in FP16 format, run the following command on the server:

```bash
python convert_llama3.1_gguf_to_torch.py --input ./Meta-Llama-3.1-8B-Instruct-Q4_K_M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --output ./Meta-Llama-3.1-8B-Instruct-Q4_K_M --just_weights
```
This will generate a dequantized weight file in FP16 format.

Output: `./Meta-Llama-3.1-8B-Instruct-Q4_K_M/pytorch.bin` (15GB)

## Step 2: Convert PyTorch Model to OpenVINO Model
After obtaining the FP16 PyTorch model, you can convert it into an OpenVINO (OV) model for optimized inference. Use the [optimum-cli](https://github.com/huggingface/optimum-intel/blob/main/docs/source/openvino/export.mdx) tool to perform the conversion:
```bash
optimum-cli export openvino --model ./Meta-Llama-3.1-8B-Instruct-Q4_K_M --weight-format int4 --group-size 128 --sym --ratio 1 ./llama3.1-8b-gguf-ov-int4 --task text-generation-with-past
```
Output: `./llama3.1-8b-gguf-ov-int4` (.bin 3.9GB)

## Step 3: Run OpenVINO GenAI pipeline

```bash
python run_ov_genai.py --model-path ./llama3.1-8b-gguf-ov-int4 --device CPU
OpenVINO version:  2024.6.0-17404-4c0f47d2335-releases/2024/6
OpenVINO is an open-source deep learning inference engine for Intel-based systems. It is a software toolkit that provides optimized performance for running deep learning models on various Intel architectures, including CPUs, GPUs, and FPGAs. OpenVINO is designed to accelerate the execution of deep learning models, making it an essential tool for applications that require high-performance inference, such as computer vision, natural language processing, and more. In this tutorial, we will cover the basics of OpenVINO, its architecture

```

### Run PyTorch pipeline
```bash
python python run_pytorch.py --model_id ./Meta-Llama-3.1-8B-Instruct-Q4_K_M --prompt "what is openvino?"
...
what is openvino? OpenVINO is an open-source software development kit (SDK) for artificial intelligence (AI) and machine learning (ML) applications. It provides a set of tools and APIs for optimizing, deploying, and managing AI models on various platforms, including CPUs, GPUs, FPGAs, and specialized AI accelerators. OpenVINO supports a wide range of frameworks and models, including TensorFlow, PyTorch, Caffe, and ONNX.
```

## Limitation
This only works for llama3.1-8b GGUF, other models might need to modify the mapping of `get_params_from_model` in `download_hf_gguf.py`
Currently not support the GGUF with .imatrix.


## Reference:
GGUF: https://github.com/huggingface/hub-docs/blob/main/docs/hub/gguf.md
gguf_reader: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py
convert gguf to torch: https://github.com/chu-tianxiang/llama-cpp-torch/blob/main/convert.py
ov genai pipeline: https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide.html
pytorch pipeline: https://huggingface.co/trinhvanhung/Meta-Llama-3.1-8B-Instruct-Q4_K_M
