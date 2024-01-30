using AbstractFFTs
using CUDA
using VkFFTCUDA
using Profile
using StatProfilerHTML

Profile.init(n=10^7, delay=0.01)

@profilehtml begin
    for i in 1:100
        x = CUDA.rand(ComplexF32, 1300)
        plan = plan_fft(x)
        res = plan * x
        res = plan \ res

        plan_tuned = plan_fft(x; tune=true)
        res = plan_tuned * x
        res = plan_tuned \ res
    end
end