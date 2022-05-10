# Benchmarks

We have tested nebullvm on popular AI models and hardware from leading vendors.

- Hardware: M1 Pro, Intel Xeon, AMD EPYC and  NVIDIA T4
- AI Models: EfficientNet, Resnet, SqueezeNet, GPT2, BERT

## Reponse time acceleration in milliseconds

The table below shows the response time in milliseconds of the non-optimized model and the optimized model for the various model-hardware couplings as an average value over 100 experiments.
  
Optimized performance is provided in the case where acceleration has not resulted in any performance loss by using deep learning compilers (Option A) or when also other otpimization techniques such as quantization and half precision (Option B) are also applied with perf_loss_ths parameter set to 2. Refer to the <a href="https://github.com/nebuly-ai/nebullvm">nebullvm library readme</a> for more clarification on the two options and the perf_loss_ths parameter.

|                   | <--   |  **M1 Pro**  |   -->      |    <--   | **Intel Xeon** |  -->      |   <--  | **AMD EPYC** |   -->     |    <--  | **Nvidia T4** |  -->     |
|:----------------------:|:-----------:|:------------:|:------------:|:-----------:|:---------------:|:------------:|:-----------:|:-------------:|:------------:|:-----------:|:-------------:|:------------:|
|                        | **Vanilla** | **Option A** | **Option B** | **Vanilla** |   **Option A**  | **Option B** | **Vanilla** |  **Option A** | **Option B** | **Vanilla** |  **Option A** | **Option B** |
| **EfficientNetB0**     |    214.95   |     24.4     |     9.24     |    36.07    |      12.15      |     10.44    |    86.29    |     38.64     |     31.67    |    12.92    |      9.59     |       -      |
| **EfficientNetB1**     |    278.81   |     33.62    |     13.28    |    50.47    |      17.33      |     16.15    |    96.65    |     59.93     |     41.69    |    17.99    |     14.19     |       -      |
| **EfficientNetB2**     |    284.88   |     36.77    |     14.56    |    50.33    |      19.06      |     17.99    |    97.32    |     65.93     |       -      |    36.91    |     13.46     |       -      |
| **EfficientNetB3**     |    370.11   |     50.37    |     20.29    |    67.98    |      26.74      |     25.83    |    207.95   |     89.61     |       -      |    20.26    |     14.33     |       -      |
| **EfficientNetB4**     |    558.86   |     70.99    |     28.03    |    91.43    |      35.89      |     35.08    |    274.93   |     119.17    |       -      |    24.89    |     17.08     |       -      |
| **EfficientNetB5**     |    704.25   |     99.84    |     41.62    |    125.69   |      53.91      |     51.7     |    481.7    |     188.63    |       -      |    31.23    |     17.94     |       -      |
| **EfficientNetB6**     |     1124    |    157.38    |     56.67    |    165.15   |      71.99      |     68.74    |    630.95   |     256.65    |       -      |    35.79    |     21.27     |       -      |
| **EfficientNetB7**     |   1521.71   |    212.12    |     81.83    |    223.15   |      106.86     |     95.85    |    766.61   |     395.57    |       -      |    45.65    |     23.32     |       -      |
| **Resnet18**           |    18.48    |     15.75    |       -      |     32.2    |      17.79      |     16.66    |    147.04   |     93.43     |     84.99    |    25.23    |     12.39     |     3.46     |
| **Resnet34**           |    42.06    |     34.4     |       -      |    61.67    |      36.54      |     33.19    |    180.18   |     166.13    |       -      |    27.41    |      5.36     |     4.79     |
| **Resnet50**           |    62.22    |     54.25    |     46.22    |     83.1    |      46.81      |     38.42    |    311.44   |     197.68    |    161.45    |     10.5    |      7.81     |     5.65     |
| **Resnet101**          |    118.95   |     92.01    |     86.48    |    152.52   |      82.99      |     71.19    |    545.65   |     364.74    |    358.55    |    20.22    |     12.82     |     9.43     |
| **Resnet152**          |    166.89   |    129.81    |    127.31    |    220.78   |      129.86     |    104.05    |    810.95   |     540.86    |       -      |    32.51    |     17.86     |     12.92    |
| **SqueezeNet**         |    15.25    |     7.86     |       -      |    23.63    |       8.7       |       -      |    86.78    |     43.49     |       -      |     3.48    |      2.7      |       -      |
| **Convnext tiny**      |    305.58   |     95.55    |     94.89    |    79.91    |      62.01      |       -      |    404.75   |     220.91    |       -      |    38.29    |      9.58     |     7.69     |
| **Convnext small**     |    615.25   |    167.78    |    167.43    |    145.05   |      110.69     |       -      |   735.037   |     544.47    |       -      |    24.31    |     17.02     |     12.21    |
| **Convnext base**      |    815.01   |     240.4    |       -      |    230.72   |      187.39     |       -      |   1237.36   |     966.58    |       -      |    76.53    |     25.79     |     15.44    |
| **Convnext large**     |   1266.87   |    394.85    |       -      |    444.82   |      396.62     |       -      |   2537.23   |    1868.43    |    1567.97   |    108.12   |     38.41     |     23.67    |
| **GPT2 - 10 tokens**   |    29.67    |     10.75    |       -      |    38.45    |      31.88      |     12.15    |    138.11   |     55.31     |     48.76    |    15.31    |      4.42     |     4.01     |
| **GPT2 - 1024 tokens** |    546.74   |       -      |       -      |   1564.67   |      924.58     |       -      |   9423.16   |    5076.11    |       -      |    84.47    |       -       |     58.63    |
| **Bert - 8 tokens**    |    39.39    |      6.2     |       -      |    31.31    |      14.87      |     10.86    |    164.9    |     38.12     |     34.08    |    10.35    |      3.78     |     2.51     |
| **Bert - 512 tokens**  |    489.52   |    276.35    |       -      |    494.21   |      376.13     |       -      |   2985.27   |    1847.31    |       -      |    31.25    |     27.37     |     10.12    |
  
</details> 
  
  
## Reponse time acceleration (inference speedup)
  
The table below displays the speedup provided by nebullvm, where speedup is defined as the response time of the optimized model over the response time of the non-optimized model. 

The speedup is shown for option A and B. We also present the B-boost, which refers to the additional acceleration provided by the techniques used only in Option B (quantization and half-precision) over those also used in Option A (deep learning compilers. Refer to the <a href="https://github.com/nebuly-ai/nebullvm">nebullvm library readme</a> for more information about Option A and B.
  
  
|                   | <--   |  **M1 Pro**  |   -->      |    <--   | **Intel Xeon** |  -->      |   <--  | **AMD EPYC** |   -->     |    <--  | **Nvidia T4** |  -->     |
|:----------------------:|:------------:|:-----------:|:------------:|:----------------:|:---------------:|:------------:|:----------------:|:-------------:|:------------:|:----------------:|:-------------:|:------------:|
|                        | **Option A** | **B-boost** | **Option B** | **DL compilers** |   **B-boost**   | **Option B** | **DL compilers** |  **B-boost**  | **Option B** | **DL compilers** |  **B-boost**  | **Option B** |
| **EfficientNetB0**     | 8.8x | 2.6x | 23.3x | 3.0x | 1.2x | 3.5x | 2.2x | 1.2x | 2.7x | 1.3x |   -  | 1.3x |
| **EfficientNetB1**     | 8.3x | 2.5x | 21.0x | 2.9x | 1.1x | 3.1x | 1.6x | 1.4x | 2.3x | 1.3x |   -  | 1.3x |
| **EfficientNetB2**     | 7.7x | 2.5x | 19.6x | 2.6x | 1.1x | 2.8x | 1.5x |   -  | 1.5x | 2.7x |   -  | 2.7x |
| **EfficientNetB3**     | 7.3x | 2.5x | 18.2x | 2.5x | 1.0x | 2.6x | 2.3x |   -  | 2.3x | 1.4x |   -  | 1.4x |
| **EfficientNetB4**     | 7.9x | 2.5x | 19.9x | 2.5x | 1.0x | 2.6x | 2.3x |   -  | 2.3x | 1.5x |   -  | 1.5x |
| **EfficientNetB5**     | 7.1x | 2.4x | 16.9x | 2.3x | 1.0x | 2.4x | 2.6x |   -  | 2.6x | 1.7x |   -  | 1.7x |
| **EfficientNetB6**     | 7.1x | 2.8x | 19.8x | 2.3x | 1.0x | 2.4x | 2.5x |   -  | 2.5x | 1.7x |   -  | 1.7x |
| **EfficientNetB7**     | 7.2x | 2.6x | 18.6x | 2.1x | 1.1x | 2.3x | 1.9x |   -  | 1.9x | 2.0x |   -  | 2.0x |
| **Resnet18**           | 1.2x |   -  |  1.2x | 1.8x | 1.1x | 1.9x | 1.6x | 1.1x | 1.7x | 2.0x | 3.6x | 7.3x |
| **Resnet34**           | 1.2x |   -  |  1.2x | 1.7x | 1.1x | 1.9x | 1.1x |   -  | 1.1x | 5.1x | 1.1x | 5.7x |
| **Resnet50**           | 1.1x | 1.2x |  1.3x | 1.8x | 1.2x | 2.2x | 1.6x | 1.2x | 1.9x | 1.3x | 1.4x | 1.9x |
| **Resnet101**          | 1.3x | 1.1x |  1.4x | 1.8x | 1.2x | 2.1x | 1.5x | 1.0x | 1.5x | 1.6x | 1.4x | 2.1x |
| **Resnet152**          | 1.3x | 1.0x |  1.3x | 1.7x | 1.2x | 2.1x | 1.5x |   -  | 1.5x | 1.8x | 1.4x | 2.5x |
| **SqueezeNet**         | 1.9x |   -  |  1.9x | 2.7x |   -  | 2.7x | 2.0x |   -  | 2.0x | 1.3x |   -  | 1.3x |
| **Convnext tiny**      | 3.2x | 1.0x |  3.2x | 1.3x |   -  | 1.3x | 1.8x |   -  | 1.8x | 4.0x | 1.2x | 5.0x |
| **Convnext small**     | 3.7x | 1.0x |  3.7x | 1.3x |   -  | 1.3x | 1.4x |   -  | 1.4x | 1.4x | 1.4x | 2.0x |
| **Convnext base**      | 3.4x |   -  |  3.4x | 1.2x |   -  | 1.2x | 1.3x |   -  | 1.3x | 3.0x | 1.7x | 5.0x |
| **Convnext large**     | 3.2x |   -  |  3.2x | 1.1x |   -  | 1.1x | 1.4x | 1.2x | 1.6x | 2.8x | 1.6x | 4.6x |
| **GPT2 - 10 tokens**   | 2.8x |   -  |  2.8x | 1.2x | 2.6x | 3.2x | 2.5x | 1.1x | 2.8x | 3.5x | 1.1x | 3.8x |
| **GPT2 - 1024 tokens** |   -  |   -  |   -   | 1.7x |   -  | 1.7x | 1.9x |   -  | 1.9x |   -  | 1.6x | 1.4x |
| **Bert - 8 tokens**    | 6.4x |   -  |  6.4x | 2.1x | 1.4x | 2.9x | 4.3x | 1.1x | 4.8x | 2.7x | 1.5x | 4.1x |
| **Bert - 512 tokens**  | 1.8x |   -  |  1.8x | 1.3x |   -  | 1.3x | 1.6x |   -  | 1.6x | 1.1x | 2.7x | 3.1x |


  
