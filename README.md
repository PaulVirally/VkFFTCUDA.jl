# VkFFTCUDA.jl
Julia bindings for [VkFFT](https://github.com/DTolm/VkFFT).

⚠️ WARNING: These bindings are very barebones and do not fully implement all the functionality exposed by VkFFT. These bindings only expose multidimensional C2C FFTs so far.

## Purpose
This package allows you to do FFTs on CuArrays using [VkFFT](https://github.com/DTolm/VkFFT) instead of cuFFT.

## Example
```julia
using CUDA, AbstractFFTs, VkFFTCUDA

x = CuArray(ComplexF32.(collect(reshape(1:60, 3, 4, 5))))
ifft(fft(x)) ≈ x # Should return true
```
