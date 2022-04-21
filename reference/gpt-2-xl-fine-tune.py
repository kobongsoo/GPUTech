import transformers

from datasets import load_dataset
from tqdm import tqdm

import bitsandbytes as bnb


model_name = 'gpt2-xl'


if __name__ == "__main__":      
    import torch

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    gpu_model = transformers.GPT2LMHeadModel.from_pretrained(model_name).half().to('cuda')
    cpu_model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    
    data = load_dataset("wikitext", "wikitext-2-v1")
    tokenizer.pad_token = tokenizer.eos_token
    test_batch = tokenizer(sorted(data['train']['text'], key=len)[-64:], max_length=1024, padding=True, pad_to_multiple_of=256, return_tensors='pt')
    test_batch['labels'] = test_batch['input_ids']
          
    N = len(test_batch['input_ids'])
    mini_batch_size = 4
    batch_size = 64
    accumulation_steps = N // mini_batch_size
    num_epochs = 20
          
    optimizer = bnb.optim.Adam8bit(cpu_model.parameters(), lr=0.001, betas=(0.9, 0.995))
          
    losses = []
    epochs = []
          
    gpu_model.train()
    cpu_model.train()
    gpu_model.gradient_checkpointing_enable()
    for i in range(10):
        epoch_losses = []
        for batch_start, batch_end in zip(range(0, N, mini_batch_size), range(mini_batch_size, N + 1, mini_batch_size)):
            input = {key: tensor[batch_start:batch_end].to('cuda') for key, tensor in test_batch.items()}
            output = gpu_model(**input, use_cache=False)
            loss = output.loss / accumulation_steps
            epoch_losses.append(loss.detach().cpu().item())
            loss.backward()
            if batch_end % batch_size == 0:
                for p_gpu, p_cpu in zip(gpu_model.parameters(), cpu_model.parameters()):
                    p_cpu.grad = p_gpu.grad.cpu().to(torch.float32)
                gpu_model.zero_grad()
                optimizer.step()
                gpu_model.load_state_dict(cpu_model.state_dict())
                cpu_model.zero_grad()
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f'Epoch {i + 1}/{num_epochs}, loss: {losses[-1]}')
