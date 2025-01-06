# FlashAttention

Flash Attention is a high-performance mechanism designed to accelerate the attention mechanism in transformers by reducing memory overhead and computation costs. This repository contains my attempts at implementing  flash attention in raw cuda.

### Executing program
run flash attention kernel with comparison to pytorch 
```console
python bench.py
```
run standalone version in standalone folder
```console
nvcc fa_1.cu
```
### Performance 


### To-do
- [x] naive implmentation of forward pass of v1
- [x] naive implmentation of forward pass of v2
- [ ] naive implmentation of backward pass of v1
- [ ] naive implmentation of backward pass of v2
- [x] optimized implementation of forward pass of v2 (coalesced memory access using chunk-based reads, optimized loop unrolling etc.)
- [ ] use warp level primitives in optimized implementation to further speed up
- [ ] add mixed precision support.
- [ ] optimized implementation of backward pass of v2
- [ ] attention masking (supports causual masking)



