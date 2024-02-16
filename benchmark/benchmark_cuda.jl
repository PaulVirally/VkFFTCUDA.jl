using BenchmarkTools
using Dates
using CUDA
using AbstractFFTs

suite = BenchmarkGroup()
suite["oop_fft"] = BenchmarkGroup()
suite["ip_fft"] = BenchmarkGroup()

sizes = 2:4096

x = CUDA.rand(ComplexF32, maximum(sizes))
for size in sizes
    view = @view x[1:size]
    oop_plan = plan_fft(view)
    ip_plan = plan_fft!(view)
    suite["oop_fft"][size] = @benchmarkable $oop_plan * $view
    suite["ip_fft"][size] = @benchmarkable $ip_plan * $view
end

tune!(suite)
results = run(suite, verbose=true)
BenchmarkTools.save("cuda_bench_" * Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS") * ".json", results)
