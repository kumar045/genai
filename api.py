# !git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
# !cd mixtral-offloading && pip install -q -r requirements.txt
# !huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo
from fastapi import FastAPI
import uvicorn
import torch
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from hqq.core.quantize import BaseQuantizeConfig
from src.build_model import OffloadConfig, QuantConfig, build_model

app = FastAPI()

# Initialize and load the model and tokenizer
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
config = AutoConfig.from_pretrained(quantized_model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

offload_per_layer = 4
num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)

attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256

ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Define a route for the text generation
@app.post("/generate/")
async def generate_text(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    attention_mask = torch.ones_like(input_ids["input_ids"])

    result = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    generated_text = tokenizer.decode(result["sequences"][0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if _name_ == "__main__":
    uvicorn.run(app, host="192.168.15.184", port=8000)

