#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino_genai as ov_genai
import openvino as ov
import argparse

        
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run OpenVINO GenAI with a specified model path.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--device", type=str, default="CPU", help="Target device to run the model on (e.g., CPU, GPU)")

    args = parser.parse_args()
    print("OpenVINO version: ", ov.get_version())

    # Initialize the pipeline with the specified model path and device
    pipe = ov_genai.LLMPipeline(args.model_path, args.device)

    # Generate text
    print(pipe.generate("what is openvino?", max_new_tokens=100))

if __name__ == "__main__":
    main()
