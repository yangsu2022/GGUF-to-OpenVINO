import os
import torch
import gguf
import numpy as np
from gguf_reader import GGUFReader

def reverse_permute(weights, n_head=32):
    return weights.reshape((n_head, weights.shape[0] // n_head // 2, 2, *weights.shape[1:])) \
                  .swapaxes(1, 2) \
                  .reshape(weights.shape)


# Modify the key name in the model file, mainly to match the key name in the Llama model
def get_params_from_model(gguf_model):
    ConstDict = {}
    ConstDict['model.embed_tokens.weight'] = gguf_model['token_embd.weight']
    ConstDict['model.norm.weight'] = gguf_model['output_norm.weight']
    ConstDict['lm_head.weight'] = gguf_model['output.weight']

    for index in range(32):
        ConstDict['model.layers.{index}.self_attn.q_proj.weight'.format(index=index)] = reverse_permute(gguf_model['blk.{index}.attn_q.weight'.format(index=index)], n_head=32)
        ConstDict['model.layers.{index}.self_attn.k_proj.weight'.format(index=index)] = reverse_permute(gguf_model['blk.{index}.attn_k.weight'.format(index=index)], n_head=8)
        ConstDict['model.layers.{index}.self_attn.v_proj.weight'.format(index=index)] = gguf_model['blk.{index}.attn_v.weight'.format(index=index)]
        ConstDict['model.layers.{index}.self_attn.o_proj.weight'.format(index=index)] = gguf_model['blk.{index}.attn_output.weight'.format(index=index)]
        ConstDict['model.layers.{index}.mlp.gate_proj.weight'.format(index=index)] = gguf_model['blk.{index}.ffn_gate.weight'.format(index=index)]
        ConstDict['model.layers.{index}.mlp.up_proj.weight'.format(index=index)] = gguf_model['blk.{index}.ffn_up.weight'.format(index=index)]
        ConstDict['model.layers.{index}.mlp.down_proj.weight'.format(index=index)] = gguf_model['blk.{index}.ffn_down.weight'.format(index=index)]
        ConstDict['model.layers.{index}.input_layernorm.weight'.format(index=index)] = gguf_model['blk.{index}.attn_norm.weight'.format(index=index)]
        ConstDict['model.layers.{index}.post_attention_layernorm.weight'.format(index=index)] = gguf_model['blk.{index}.ffn_norm.weight'.format(index=index)]

    return ConstDict

def convert_to_state_dict(checkpoint, save_dir, just_weights=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = {}
    result = GGUFReader(checkpoint)
    architecture = result.fields['general.architecture']
    architecture = str(bytes(architecture.parts[architecture.data[0]]), encoding = 'utf-8')

    model_name = str(bytes(result.fields['general.name'].parts[result.fields['general.name'].data[0]]), encoding = 'utf-8')

    if architecture not in ["llama", "qwen2", "internlm2", "starcoder2", "qwen",
                            "stablelm", "orion", "minicpm", "gemma", "xverse", "command-r"]:
        print(f"Unsupported architecture {architecture}")
        return
    # write tensor
    for ts in result.tensors:
        if hasattr(ts.data.dtype, 'names') and ts.data.dtype.names is not None:
            for name in ts.data.dtype.names:
                state_dict[ts.name + "_" + name] = torch.tensor(ts.data[name])
        else:
            if ts.tensor_type in gguf.quants._type_traits.keys():
                q_weights = torch.tensor(ts.data).numpy()
                q_weights_c = q_weights.copy(order="C")
                dq_weights = gguf.quants.dequantize(q_weights_c, ts.tensor_type.value)
                if dq_weights.sum() != 0.0:
                    state_dict[ts.name] = torch.tensor(dq_weights.astype(np.float16))
            else:
                if ts.data.sum() != 0.0:
                    state_dict[ts.name] = torch.tensor(ts.data.astype(np.float16))

    dicts = get_params_from_model(state_dict)
    torch.save(dicts, os.path.join(save_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert GGUF checkpoints to torch')

    parser.add_argument('--input', type=str, default='./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', help='The path to GGUF file')
    parser.add_argument('--output', type=str, default='./test', help='The path to output directory')
    parser.add_argument('--just_weights', action="store_true", help='just convert weights.')
    args = parser.parse_args()
    convert_to_state_dict(args.input, args.output, args.just_weights)
