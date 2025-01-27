# Multiple-Model Coding (MMC) Scheme for Electrical Signal Compression with Quality Constraint

> **Citation:**
>
> @article{PresvotsMMC2023  
> author = {Presvôts, Corentin and Kieffer, Michel and Prevost, Thibault and Panciatici, Patrick and Li, Zuxing and Piantanida, Pablo},  
> title = {Multiple-Model Coding Scheme for Electrical Signal Compression},  
> year = {2023},  
> note = {Available at SSRN: [https://ssrn.com/abstract=4584757](https://ssrn.com/abstract=4584757) or [http://dx.doi.org/10.2139/ssrn.4584757](http://dx.doi.org/10.2139/ssrn.4584757)}  
> }

This code proposes a low-latency Multiple-Model Coding approach to compress sampled electrical signal
waveforms under encoding rate constraints. The approach is window-based. Several parametric waveform models
are put in competition to obtain a first coarse representation of the signal in each considered window. Then, different
residual compression techniques are compared to minimize the residual reconstruction error. The model parameters
are quantized, and the allocation of the rate budget among the two steps is optimized.

Article is available at: [MMC](https://www.researchgate.net/publication/374226674_Multiple-Model_Coding_Scheme_for_Electrical_Signal_Compression)
You can integrate the two tables into your README as follows:

---

## Stage 1: The various competing models include

### Model types and their distributions
The following table outlines the model types used in the first stage of the Multiple-Model Coding (MMC) scheme, along with the corresponding a priori distributions for their parameters:

| Model type                  | $p_{\boldsymbol{\theta}^{m}}$                                                                                               |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Bypass**                 |                                                                                                                             |
| **Sinusoidal**              | $\mathcal{U}\left(\boldsymbol{\theta}^m;(0.5,40.9,-\pi)^T,(1,50.1,\pi)^T\right)$ |
| **Polynomial, order $K$ with $K\in\left[0,\dots,9\right]$**  | $\mathcal{U}\left(\boldsymbol{\theta}^m;-(\frac{w_1^m}{2}, \dots,\frac{w_{K+1}^m}{2}))^T,(\frac{w_1^m}{2}, \dots, \frac{w_{K+1}^m}{2})^T\right)$ |
| **Sample predictive, order $K$ with $K\in\left[1,2\right]$, $\eta=\in\left[0,1\right]$**| $\mathcal{U}\left(\boldsymbol{\theta}^m;-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$                      |
| **Parameter predictive**        | $\mathcal{U}\left(\boldsymbol{\delta}\boldsymbol{\theta}^m;-0.1\cdot\boldsymbol{1},0.1\cdot\boldsymbol{1}\right)$          |

In these distributions:
- $\mathcal{U}(\boldsymbol{\theta}^m; \boldsymbol{a}^m, \boldsymbol{b}^m)$ represents a uniform distribution with parameters $\boldsymbol{a}^m$ and $\boldsymbol{b}^m$.
- $\boldsymbol{1}$ is a vector of ones matching the dimension of $\boldsymbol{\theta}^m$.
- For the polynomial model, we choose uniform prior distributions with widths $w_k^m$ for the $k$-th coefficient of the polynomial model $m$. The values of $w_k^m$ were determined to encapsulate 90\% of the total energy of the distribution of $\theta_k^m$, based on an analysis of 20,000 signals from the [Data\_u](https://github.com/rte-france/digital-fault-recording-database).


### Stage 2: Residual compression methods
The second stage of the MMC scheme applies various residual compression methods, each with an associated a priori distribution:

| Method type                | $p_{\boldsymbol{y}^{\ell}}$                                                   |
|----------------------------|-------------------------------------------------------------------------------|
| **Bypass**                 |                                                                               |
| **[DCT + BPC](https://www.academia.edu/26719905/A_new_bitplane_coder_for_scalable_transform_audio_coding)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |
| **[DWT + BPC](https://ieeexplore.ieee.org/document/6682511)**              | $\mathcal{U}\left(\boldsymbol{y};-\frac{1}{2}\boldsymbol{1},\frac{1}{2}\boldsymbol{1}\right)$ |


Refer to the detailed sections in the paper for further explanation of each method’s parameters and performance considerations.

---

## Explanation of `main.py`

The main code handles the compression process for each window by:  
1. Using the selected encoding method to compress the samples within a window.  
2. Producing a binary frame that represents the compressed signal.  
3. The decoder then reconstructs the original signal from this binary frame.

`main.py` is designed to encode selected voltage and current signals using a multiple-model coding (MMC) approach. The signals and various parameters can be adjusted directly within the script to tailor the encoding process.

---

### Key Parameters

#### Data Source and Parameters 
By default, the code uses 12 three-phase voltage signals from the [Data_S](https://github.com/rte-france/digital-fault-recording-database) dataset. These signals are selected because they correspond to known faults. Each signal is one second long and sampled at 6400 Hz, resulting in 100 non-overlapping 20 ms windows per signal.

- **Number of signals:** The number of signals to encode is controlled by `nb_signal`.  
- **Number of phases:** Specified by `nb_phase`:  
  - `nb_phase=3` processes only the three voltage phases.  
  - `nb_phase=6` includes both the three voltage phases and three current phases for all 12 signals.  
- **Number of windows (`nb_w`):** By default, the first 50 windows of each signal are encoded. This can be adjusted to process more or fewer windows per signal.
- **Window Size (`N`):** Each window is set to 128 samples by default. This can be adjusted by modifying `N` in the script.  

#### For Quality Constraint
- **Quality Constraint (quality):** The encoding quality for each window is defined by a default RMSE of 200 V. You can adjust quality to experiment with different levels of compression performance.
- **Available Metrics:** Three metrics derived from the L2 norm are implemented: RMSE, MSE, and -SNR.


#### For Model Compression
- The set of models ($\mathcal{M}$) is initialized in the main code. You can exclude certain medels by commenting the coesponding lines in the main code.
- It is also possible to add polynomial models of degree 0 to 14, as well as sample predictive models with different orders and values for $\eta$.
- You can also consider other models by modifying the initialization in the main code, particularly by adjusting the values in $p_{\boldsymbol{\theta}^{m}}$ to reflect different *a-priori* distributions.

#### For Residual Compression Methods
- The set of residual compression methods ($\mathcal{L}$) are also initialized in the main code. You can exclude certain methods by commenting the corresponding lines in the main code.

---

### Compression Techniques
There are multiple encoding methods available in `main.py`, each offering a different balance between computational complexity and compression performance. Depending on your computational resources and the desired compression quality, you can select one of the following methods:

1. **Exhaustive Search:**  
For each $n_{\text{x}}$, the model parameters are quantized, and all residual compression methods are tested until a distortion constraint is met. The best combination of parameter quantization and residual compression is selected. This approach is the most computationally expensive but achieves the best compression performance.
   - High computational cost (~100×O(Nlog(N))).  
   - Example: 92.8 bits/window for the 12 signals with RMSE=200V.

2. **Golden Section Search:**  
This method assumes convexity in the bit allocation problem and uses the golden section search to iteratively narrow the range of $n_{\text{x}}$. It requires fewer function evaluations than exhaustive search, resulting in faster convergence, though with slightly reduced compression quality.
   - Mid-level computational complexity (~74.2×O(Nlog(N))).  
   - Slightly lower compression performance than exhaustive search, 92.6 bits/window for the 12 signals with RMSE=200V.

3. **Sub-optimal Exhaustive Search:**  
This process starts by selecting $n_{\text{x}}$ values and first determining the best model. For the best model, the best residual compression method is then identified. This simplified exhaustive search evaluates fewer configurations, reducing complexity while maintaining reasonable compression performance.
   - Reduced complexity (~34.5×O(Nlog(N))).  
   - 91.7 bits/window for the 12 signals with RMSE=200V.

4. **Distortion Model Exhaustive Search:**  
A distortion model is used to identify a subset of candidate models and bit allocations. An exhaustive search is then performed on this reduced set to find the optimal $n_{\text{x}}$, resulting in significant complexity reduction at a modest cost to compression quality.
   - Further reduces complexity (~16×O(Nlog(N))).  
   - 94.6 bits/window for the 12 signals with RMSE=200V.

5. **Distortion Model Golden Section Search:**  
This approach applies a distortion model to pre-select candidate models and bit allocations, followed by a golden section search to find the optimal $n_{\text{x}}$. It is the least computationally demanding method, trading off some compression performance for speed.
   - The least complex method (~10.5×O(Nlog(N))).  
   - Fastest option, but with the lowest compression performance : 92=6.9 bits/window for the 12 signals with RMSE=200V..

For 4 and 5: You can further reduce the complexity of these approaches by adjusting certain parameters in the code. Specifically, modifying self.delta_M (= 3 ini) (the number of top-performing models retained for rate-distortion model predictions) and self.delta_nx (= 4 ini) (which sets the search interval around the distortion model’s predicted optimal $n_{\text{x}}$) can narrow the search space. By selecting smaller values for these parameters, the code can focus on fewer candidates and narrower intervals, leading to faster computations at the expense of possibly skipping some alternative configurations.

---

# Prerequisites

- numpy


- matplotlib.pyplot


- accumulate from the itertools library


- dct, idct from the scipy.fftpack library


- pywt


- fsolve from the scipy.optimize library
