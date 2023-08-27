# Deepspeed

## ZeRO (Zero Redundancy Optimizer)
和数据并行一起使用的显存优化方法，不用改模型代码(改的少)

**partition**
Stage 1: 切分优化器状态，每个process（gpu）只分到一部分优化器状态也只更新这一部分优化器状态

Stage 2: 切分优化器状态和梯度，每个process只保存自己的优化器状态和对应的梯度

Stage 3: 切分优化器状态，梯度和模型参数，每个process分到一部分模型参数，也只保存对应的优化器状态和梯度

    这和模型并行有什么区别：模型并行是把模型切分到每张卡上，计算前Fetch activation；stage3是把计算前Fetch model weight，然后整合成完整的layer再计算

**offload**

stage2/3可以，通过offload_optimizer/offload_param来offload到cpu上；zero-offload是stage2的升级版


## 3D Parallelism
tensor-slicing, pipeline-parallelism, and data parallelism

- deepspeed支持tensor并行吗？
    
    deepspeed-megatron可以tensor并行，但是实际上是megatron做tensor并行，deepspeed配合一下
    
    deepspeed可以对一些huggingface的模型做推理时的tensor并行
    
    zero-stage3支持tensor-slicing，能算tensor并行吗？（这个是怎么做的呢？）
    
        -> 原生deepspeed并不支持tensor并行训练

- pipeline并行
    
    - 模型：
        
        模型需要表示成layer列表（nn.Sequential）
    
        forward默认就是每层输出作为下一层输入，对输入输出形式提出了要求: 必须是tensor或者tuple of tensors
    
        net = PipelineModule(layers=layer_sequential_net, num_stages=pipeline_stage_number)
    
        还需要传入loss_fn
    
    - 训练：
    
        loss = engine.train_batch(data_iter=train_iter)，train_batch内部包含了forward/backward/step，因为pipeline并行，这三步在不同GPU之间交替进行，所以无法单独写出来
        
        -> 也就是说engine的最后需要计算好loss，也就是需要传入loss_fn
        
        dataloader需要提供：input和label
        
        每个train_batch中，dataloader会被调engine.gradient_accumulation_steps()次，每次dataloader会提供size为engine.train_micro_batch_size_per_gpu()的数据
        
            也就是说，一个train_batch里面真正被训练的数据有gradient_accumulation_steps*train_micro_batch_size_per_gpu条
        
        train_batch中途数据不能中断（为空），用deepspeed.utils.RepeatingLoader来重复dataloader保证这一点
    
    - NOTE:
    
        - 用list of LayerSpec替代nn.Sequential：
    
            如果用原生pytorch的方式来load模型，在初始化的时候每个worker都需要把整个模型放进cpu，需要的cpu memory就是work数量*模型大小->这谁顶得住啊
    
            LayerSpec可以让每个worker只申请自己需要的模型层，cpu memory就是模型大小
    
        - TiedLayerSpec：处理embedding在LM中存在tie的情况

        - batch-size：
        
            chunks/gradient accumulation steps: 每个pipeline stage（一个gpu）会forward多次，再backward多次，以减小PP中的bubble

            因此，一个gpu实际一次forward的batch-size（micro-batche-size）= 全局bs / 数据并行数 /chunks

参考:

[Training your large model with DeepSpeed](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)

[ZeRO and model parallelism](https://github.com/microsoft/DeepSpeed/discussions/1911)

[DeepSpeed - Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
