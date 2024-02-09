# VkFFTCUDA.jl

This package provides julia bindings to [VkFFT](https://github.com/DTolm/VkFFT).

## Installation
First, make sure you have CUDA and CMake installed on your system. These are required dependencies. After that, installation is very simple. Run the following in a julia REPL:
```julia
import Pkg
Pkg.add("VkFFTCUDA")
```

You should now be able to use VkFFTCUDA.jl:
```julia
using CUDA, AbstractFFTs, VkFFTCUDA

x = CuArray(ComplexF32.(collect(reshape(1:60, 3, 4, 5))))
ifft(fft(x)) ≈ x # Should return true
```

Since VkFFTCUDA implements the API provided by [AbstractFFTS.jl](https://github.com/JuliaMath/AbstractFFTs.jl), you should refer to [their documentation](https://juliamath.github.io/AbstractFFTs.jl/stable/) to learn about all the available functionality.
