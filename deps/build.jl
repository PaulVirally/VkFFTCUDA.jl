# Check if we have nvcc
has_nvcc = false
try
    run(`which nvcc`)
    global has_nvcc = true
catch
    global has_nvcc = false
end

if has_nvcc
    tmp = Base.Filesystem.tempdir() * "/"
    pwd = Base.Filesystem.pwd()
    clone_cmd = Cmd(`git clone --recurse-submodules https://github.com/PaulVirally/VkFFTCUDALib`, dir=tmp)
    mkbuilddir_cmd = Cmd(`mkdir build`, dir=tmp*"/VkFFTCUDALib")
    cmake_cmd = Cmd(`cmake -DCMAKE_BUILD_TYPE=Release ..`, dir=tmp*"/VkFFTCUDALib/build")
    make_cmd = Cmd(`make`, dir=tmp*"/VkFFTCUDALib/build")
    makeinstall_cmd = Cmd(`sudo make install`, dir=tmp*"/VkFFTCUDALib/build")
    rmtmp_cmd = Cmd(`rm -rf VkFFTCUDALib`, dir=tmp)

    run(clone_cmd)
    run(mkbuilddir_cmd)
    run(cmake_cmd)
    run(make_cmd)
    run(makeinstall_cmd)
    run(rmtmp_cmd)
else
    println("nvcc not found. Please install CUDA to use VkFFTCUDA.jl")
end