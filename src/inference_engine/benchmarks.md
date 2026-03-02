
Benchmarks for Qwen models with 16k seq len, ran on a tpu-v58

> **TPS**: Tokens Per Second  -- Total Throughput
> **SPS**: Samples Per Second -- How many steps can you take in a second across the whole batch size
> **Total Time**: Time for total completion 


| Model Name | n_replicas | batch_size | TPS | SPS | Total Time | 
|------------|------------|------------|-----|-----| ---------- | 
| Qwen 0.6B  | 1          | 1          |     |     |  
| Qwen 0.6B  | 4          | 1          |     |     |
| Qwen 0.6B  | 4          | 16         |     |     |
| Qwen 1.7B  | 1          | 1          |     |     |
| Qwen 1.7B  | 4          | 1          |     |     |
| Qwen 1.7B  | 4          | 16         |     |     |
| Qwen 4B    | 1          | 1          |     |     |
| Qwen 4B    | 4          | 1          |     |     |
| Qwen 4B    | 4          | 16         |     |     |
