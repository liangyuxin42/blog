# Megatron-LM
- 数据加载
    
    预先 tokenize 和 shuffle

- Fused CUDA Kernels

    把torch中的多个操作融合成一个，降低数据在内存和gpu之间来回移动的次数

- 环境：

    pytorch、cuda、nccl、NVIDIA APEX 和 nltk 库（并且合理地根据你的显卡环境配置版本，which can be tricky）

    然后 git clone https://github.com/NVIDIA/Megatron-LM

- 核心：tensor并行

    核心的核心：

    - 模型可以被拆分到矩阵乘法+非线性层

    - 矩阵乘法可以被横切或竖切，只需要在输入或输出同步一次即可

        Note: 每个线性每次计算输入输出都要同步，因此Tensor并行只适合GPU之间通信很快的情况，不推荐在节点之间做TP

    * 新进展：加入了sequence parallelism，把不能做tensor并行的操作按sequence方向切分


Good Read:
[The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed)
[Megatron Paper](https://arxiv.org/abs/2104.04473)