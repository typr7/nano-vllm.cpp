# nano-vllm.cpp
本项目为 [NanoInfer](https://github.com/typr7/NanoInfer) 的重构项目，实现边界如下：
1. Linux 86_64, 单 NV GPU 推理
2. C++17, CUDA 13.0, CMake 工程
3. 模型支持 qwen3-0.6B, llama3.2-1B，Hugging Face Safetensors 读取
4. tokenizer 使用 Hugging Face 提供的 tokenizer
5. 覆盖的模型算子：Embedding, RMSNorm, RoPE, GQA (FlashAttention), Q/K Norm, SwiGLU MLP, LM Head, Projection (GEMM)
7. 采样支持：greedy, temperature, top-k, top-p
8. 推理框架的前后端分离，前端 (Python) 负责 api serving, tokenizer, chat template, 模型拉取，后端 (C++) 负责原 vLLM Engine Core 的工作。前后端通过 zmq 通信。
9. 实现 PagedAttention, Continuous Batching, Chunked Prefill, Prefix Caching