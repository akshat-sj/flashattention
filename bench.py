import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_dot_product_attention


minimal_attn = load(
    name='minimal_attn',
    sources=['src/main.cpp', 'src/test.cu'],
    extra_cuda_cflags=['-O2', '-allow-unsupported-compiler']
)


batch_size = 8
n_head = 16
seq_len = 1024
head_embd = 32

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print(q.device, k.device, v.device)

print('=== Profiling Manual Attention ===')


def manual_attn(q, k, v, head_embd):
    S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_embd)  
    A = F.softmax(S, dim=-1) 
    O = torch.matmul(A, v)  
    return O

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v, head_embd)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== Profiling Minimal Flash Attention ===')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== Profiling PyTorch Flash Attention ===')


def pytorch_flash_attn(q, k, v, head_embd):
    return scaled_dot_product_attention(q, k, v)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    flash_result = pytorch_flash_attn(q, k, v, head_embd)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== Comparing Results ===')


print('Manual vs Minimal:', torch.allclose(manual_result, minimal_result, rtol=0, atol=1e-02))
print('Manual vs Flash:', torch.allclose(manual_result, flash_result, rtol=0, atol=1e-02))
print('Minimal vs Flash:', torch.allclose(minimal_result, flash_result, rtol=0, atol=1e-02))
