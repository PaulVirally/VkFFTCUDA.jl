using CUDA, AbstractFFTs, VkFFTCUDA, Test

AbstractFFTs.TestUtils.test_complex_ffts(CuArray{ComplexF32}, test_inplace=true, test_adjoint=false)
AbstractFFTs.TestUtils.test_complex_ffts(CuArray{ComplexF64}, test_inplace=true, test_adjoint=false)
