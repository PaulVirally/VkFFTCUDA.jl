tmp = Base.Filesystem.tempdir() * "/"
pwd = Base.Filesystem.pwd()
clone_cmd = Cmd(`git clone --recurse-submodules https://github.com/PaulVirally/VkFFTCUDALib`, dir=tmp)
mkbuilddir_cmd = Cmd(`mkdir build`, dir=tmp*"/VkFFTCUDALib")
cmake_cmd = Cmd(`cmake -DCMAKE_BUILD_TYPE=Release ..`, dir=tmp*"/VkFFTCUDALib/build")
make_cmd = Cmd(`make`, dir=tmp*"/VkFFTCUDALib/build")
makeinstall_cmd = Cmd(`sudo make install`, dir=tmp*"/VkFFTCUDALib/build")

run(clone_cmd)
run(mkbuilddir_cmd)
run(cmake_cmd)
run(make_cmd)
run(makeinstall_cmd)
