# DeepSeek-VL2

**[DeepSeek-VL2]((https://github.com/deepseek-ai/DeepSeek-VL2))** is a powerful, efficient, and open-source Vision-Language MoE Mixture-of-Experts model.

For more details, please visit our [[Paper]](https://arxiv.org/abs/2403.08811) [[Project Page]](https://deepseek-vl2.github.io/) [[Demo]](https://deepseekvl2-demo.hf.space) [[Chat]](https://platform.deepseek.com/) 

<p align="center">
    <img src="assets/deepseek-vl2.png" width="450px">
</p>

## Latest News
* **[2024-03]** 🔥 We release **DeepSeek-VL2** series, including DeepSeek-VL2-Tiny, DeepSeek-VL2-Small, and DeepSeek-VL2-base. 
* **[2024-02]** We release **DeepSeek-VL** series, including DeepSeek-VL-7B-base and DeepSeek-VL-7B-chat.

## Introduction
DeepSeek-VL2, an advanced large Vision-Language Model (LVLM) built upon DeepSeek's Mixture-of-Experts (MoE) architecture, delivers superior performance across a wide range of multimodal benchmarks while maintaining high efficiency. This version introduces several key improvements:

1. **Innovative Mixture-of-Experts Architecture**: We integrate the MoE architecture into the vision-language domain, sparsely activating parameters to enhance the model's capability while managing computational costs.
2. **High-Resolution Image Understanding**: With a dynamic resolution mechanism and high-quality image encoders, DeepSeek-VL2 can process images at up to 1120x1120 pixels, ensuring detailed visual understanding across diverse domains.
3. **Strong Video Understanding**: DeepSeek-VL2 demonstrates robust video comprehension capabilities, particularly excelling in high-frame-rate, high-resolution video analysis.
4. **Multilingual Support**: Beyond English and Chinese, we expand the model's multilingual capabilities to include a broader range of languages, enhancing accessibility for users worldwide.

## Model Zoo
| Model | #激活参数 | Context Length | Download |
|-------| -----| ----- | ---- |
| DeepSeek-VL2-Tiny | ~1B | 4K | [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) |
| DeepSeek-VL2-Small | ~3B | 4K | [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2-small) |
| DeepSeek-VL2 | ~16B | 4K | [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2) |

For the old version, please check [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL).

## Quick Start

### Environment Setup
```shell
conda create -n deepseek-vl2 python=3.10
conda activate deepseek-vl2
pip install -r requirements.txt
```

### Inference
We provide an example to run inference with a single image:
```python
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# single image inference
conversation = [
    {
        "role": "<|User|>",
        "content": "<|image_1|>\nCan you describe the content of this image?",
        "images": ["images/cat.jpg"]
    },
    {"role": "<|Assistant|>", "content": ""}
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run the model to get the response
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

We also provide an example to run inference with multiple images/interleaved image-text:
```python
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# multiple images/interleaved image-text
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <|image_1|>\nThis is image_2: <|image_2|>\nThis is image_3: <|image_3|>\n Can you tell me what are in the images?",
        "images": [
            "images/multi_image_1.jpeg",
            "images/multi_image_2.jpeg",
            "images/multi_image_3.jpeg",
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run the model to get the response
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

We also provide an example to run inference with a video:
```python
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Video understanding
conversation = [
    {
        "role": "<|User|>",
        "content": "<|video_1|>\nCan you describe the content of this video?",
        "videos": ["videos/sample_video.mp4"]
    },
    {"role": "<|Assistant|>", "content": ""}
]

# load videos and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run the model to get the response
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

We also provide an example to run inference with incremental prefilling to reduce GPU memory usage:
```python
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# multiple images/interleaved image-text
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <|image_1|>\nThis is image_2: <|image_2|>\nThis is image_3: <|image_3|>\n Can you tell me what are in the images?",
        "images": [
            "images/multi_image_1.jpeg",
            "images/multi_image_2.jpeg",
            "images/multi_image_3.jpeg",
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# incremental prefilling when using limited GPU memory
inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
    input_ids=prepare_inputs.input_ids,
    images=prepare_inputs.images,
    images_seq_mask=prepare_inputs.images_seq_mask,
    images_spatial_crop=prepare_inputs.images_spatial_crop,
    attention_mask=prepare_inputs.attention_mask,
    chunk_size=512 # prefilling size
)

# run the model to get the response
outputs = vl_gpt.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
    past_key_values=past_key_values
)

answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
```

### Gradio Web Demo
To launch a web demo locally:
```shell
python web_demo.py --model_name deepseek-ai/deepseek-vl2-tiny
```

### Fine-tuning
We provide an example to run fine-tuning on a single GPU:
```shell
torchrun --nproc_per_node=1 finetune.py \
    --model_path deepseek-ai/deepseek-vl2-tiny \
    --data_path /path/to/training_data \
    --output_dir /path/to/output_dir
```

We also provide an example to run fine-tuning on multiple GPUs:
```shell
torchrun --nproc_per_node=8 finetune.py \
    --model_path deepseek-ai/deepseek-vl2-tiny \
    --data_path /path/to/training_data \
    --output_dir /path/to/output_dir
```

We also provide an example to run fine-tuning on multiple nodes:
```bash
# node 0
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.1.1 --master_port=12345 finetune.py \
    --model_path deepseek-ai/deepseek-vl2-tiny \
    --data_path /path/to/training_data \
    --output_dir /path/to/output_dir

# node 1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=192.168.1.1 --master_port=12345 finetune.py \
    --model_path deepseek-ai/deepseek-vl2-tiny \
    --data_path /path/to/training_data \
    --output_dir /path/to/output_dir
```

## Disclaimer
The weights for the DeepSeek-VL2 series are available for commercial use. However, please note that we are not responsible for any copyright or legal issues arising from the use of these weights. It is the user's responsibility to ensure compliance with all applicable laws and regulations.

## License
This code is licensed under the [Apache License 2.0](LICENSE). The use of the DeepSeek-VL2 model weights is subject to the [DeepSeek License](MODEL_LICENSE).

## Acknowledgements
The code is based on [Mistral](https://github.com/mistralai/Mistral), [Qwen](https://github.com/QwenLM/Qwen), and [Transformers](https://github.com/huggingface/transformers). Thanks for the open-sourcing!

1. related project [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
2. related project [Aria](https://github.com/rhymes-ai/Aria)
