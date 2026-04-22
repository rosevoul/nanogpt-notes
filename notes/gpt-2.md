# GPT-2
**Tensorflow vs Pytorch**  
- The original code from the GPT-2 paper is written in tensorflow, we are using pytorch here.
- Here is the huggingface implementation, also using pytorch, but a bit more complicated https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

**Model parameters**  
- We are reproducing the 124M parameter model, not the 1.5B one (gpt2-xl)

**Model components**  
- wte: tokens lookup table, we have 50257 tokens and each is of embedding size of 768
- wpe: positions lookup table, we have 1024 positions that each token can be attending to in the past, the position vector embedding is of size 768 as well (this is learned by the optimization)
