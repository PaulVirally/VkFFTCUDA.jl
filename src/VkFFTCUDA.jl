module VkFFTCUDA

__precompile__(false) # Apparently because we are overwriting stuff from CUDA.jl, __precompile__ must be false to avoid annoying warnings

export VkFFTPlan

using LinearAlgebra
using AbstractFFTs
import CUDA: CuPtr, CuArray

const libvkfft = "/usr/local/lib/libVkFFTCUDA.so"

try
    global VKFFT_MAX_FFT_DIMENSIONS = @ccall libvkfft.max_fft_dimensions()::Culonglong # The maximum number of dimensions that VkFFT supports (set at compile time, currently 4)
catch
    # We do not have VkFFT installed
end

"""
    VkFFTPlan{T}(app::Ptr{Cvoid}, config::Ptr{Cvoid}, direction::Int32, dims::NTuple{N, Int} where N, region, buffer_size::AbstractArray{Int}, type::DataType, is_inplace::Bool)

A struct that holds the plan for a VkFFT operation.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT
- `config::Ptr{Cvoid}`: A struct that holds the VkFFFT configuration
- `direction::Int32`: -1 for forward, 1 for inverse
- `dims::NTuple{N, Int} where N`: The size of the input array
- `region`: The dimensions on which to perform the FFT
- `buffer_size::AbstractArray{Int}`: The size of the buffer VkFFT will use (in bytes)
- `is_inplace::Bool`: Whether the plan is in-place (for optimization)
"""
mutable struct VkFFTPlan{T <: Union{ComplexF32, ComplexF64}} <: AbstractFFTs.Plan{T}
    app::Ptr{Cvoid} # The equivalent of an FFTW plan but for VkFFT
    config::Ptr{Cvoid} # A struct that holds all of the config for VkFFT
    direction::Int32 # -1 for forward, 1 for inverse
    dims::NTuple{N, Int} where N # The dimensions of the input array
    region
    buffer_size::AbstractArray{Int}
    is_inplace::Bool
    pinv::AbstractFFTs.Plan # Used to cache the inverse plan

    function VkFFTPlan(app::Ptr{Cvoid}, config::Ptr{Cvoid}, direction::Int32, dims::NTuple{N, Int} where N, region, buffer_size::AbstractArray{Int}, type::DataType, is_inplace::Bool)
        plan = new{type}(app, config, direction, dims, region, buffer_size, is_inplace)
        finalizer(plan) do plan
            try
                pinv = plan.pinv
                if pinv isa VkFFTPlan
                    _delete_plan(pinv)
                end
            catch e
                if !(e isa UndefRefError)
                    rethrow(e)
                end
            end
            _delete_plan(plan)
        end
        return plan
    end
end

"""
    _make_plan(x::CuArray{T}, region, forward::Bool, inplace::Bool; coalesced_memory::Int=0, aim_threads::Int=0, num_shared_banks::Int=0)

Creates a VkFFT plan for the given input array.

# Arguments
- `x::CuArray{T}`: The input array
- `region`: The dimensions on which to perform the FFT
- `forward::Bool`: Whether to perform a forward FFT
- `inplace::Bool`: Whether to perform the FFT in-place
- `coalesced_memory::Int=0`: The number of coalesced memory banks to use (0 for automatic)
- `aim_threads::Int=0`: The number of threads to aim for (0 for automatic)
- `num_shared_banks::Int=0`: The number of shared memory banks to use (0 for automatic)

# Returns
- A VkFFTPlan
"""
function _make_plan(x::CuArray{T}, region, forward::Bool, inplace::Bool; coalesced_memory::Int=0, aim_threads::Int=0, num_shared_banks::Int=0) where T <: Union{ComplexF32, ComplexF64}
    num_buffer_dim = ndims(x)

    dim_sizes = zeros(Int, VKFFT_MAX_FFT_DIMENSIONS)
    dim_sizes[1:num_buffer_dim] .= size(x)

    dims_to_omit = setdiff(1:num_buffer_dim, region)
    omit_dims = zeros(Int, VKFFT_MAX_FFT_DIMENSIONS)
    omit_dims[dims_to_omit] .= 1

    num_fft_dim = 3 - omit_dims[3] * (omit_dims[2] + 1) # This works for VKFFT_MAX_FFT_DIMENSIONS = 4

    num_batches = prod(size(x)[collect(dims_to_omit)])

    double_precision = eltype(x) <: ComplexF64

    config = _new_config(num_fft_dim, num_buffer_dim, dim_sizes, omit_dims, num_batches, coalesced_memory, aim_threads, num_shared_banks, forward, double_precision, inplace)
    if config == C_NULL
        throw(ArgumentError("Fatal VkFFT error: Failed to create config. Was CUDA initialized?"))
    end

    res_ptr = Ref{Culonglong}(0)
    app = _new_app(config, res_ptr) 
    if app == C_NULL
        @ccall libvkfft.delete_config(config::Ptr{Cvoid})::Cvoid
        throw(ArgumentError("Fatal VkFFT error: Failed to create app"))
    end
    if res_ptr[] != 0
        @ccall libvkfft.delete_config(config::Ptr{Cvoid})::Cvoid
        err = _vkffterr2string(res_ptr[])
        throw(ArgumentError("Fatal VkFFT error: $err (code: $(res_ptr[]))"))
    end

    direction = forward ? Int32(-1) : Int32(1)
    return VkFFTPlan(app, config, direction, size(x), region, dim_sizes, eltype(x), inplace)
end

"""
    _delete_plan(plan::VkFFTPlan)

Deletes a VkFFT plan.

Wrapper for the VkFFT C functions delete_app and delete_config.
"""
function _delete_plan(plan::VkFFTPlan)
    @ccall libvkfft.delete_app(plan.app::Ptr{Cvoid})::Cvoid
    @ccall libvkfft.delete_config(plan.config::Ptr{Cvoid})::Cvoid
end

"""
    _get_tuned_params(x::CuArray{T}, region, forward::Bool, inplace::Bool; num_ffts::Int=1000)

Tunes the parameters for a VkFFT plan.

# Arguments
- `x::CuArray{T}`: The input array
- `region`: The dimensions on which to perform the FFT
- `forward::Bool`: Whether to perform a forward FFT
- `inplace::Bool`: Whether to perform the FFT in-place
- `num_ffts::Int=1000`: The number of FFTs to run to determine the best parameters

# Returns
- A tuple of the best parameters
"""
function _get_tuned_params(x::CuArray{T}, region, forward::Bool, inplace::Bool; num_ffts::Int=1000) where T <: Union{ComplexF32, ComplexF64}
    # The set of parameters to try
    coalesced_memory_vals = [32, 64, 128]
    aim_threads_vals = [32, 64, 128, 256]

    params = Iterators.product(coalesced_memory_vals, aim_threads_vals)
    best_time = Inf
    best_params = (0, 0, 0)
    for (coalesced_memory, aim_threads) in params
        plan = _make_plan(x, region, forward, inplace; coalesced_memory=coalesced_memory, aim_threads=aim_threads)

        # Run once to warm up (BenchmarkTools.jl does this too)
        res = plan * x

        # Run num_ffts times and take the best time
        min_time = Inf # I think using the best run is a good way to avoid slow outliers
        for _ in 1:num_ffts
            time = @elapsed res = plan * x # Make sure the expression is run and not optimized away by assigning its output to res
            min_time = min(min_time, time)
        end

        if min_time < best_time
            best_time = min_time
            best_params = (coalesced_memory, aim_threads)
        end
    end
    return best_params
end

"""
    _make_tuned_plan(x::CuArray{T}, region, forward::Bool, inplace::Bool; tune::Bool=false)

Creates a tuned VkFFT plan for the given input array.

# Arguments
- `x::CuArray{T}`: The input array
- `region`: The dimensions on which to perform the FFT
- `forward::Bool`: Whether to perform a forward FFT
- `inplace::Bool`: Whether to perform the FFT in-place
- `tune::Bool=false`: Whether to tune the parameters

# Returns
- A VkFFTPlan
"""
function _make_tuned_plan(x::CuArray{T}, region, forward::Bool, inplace::Bool; tune::Bool=false) where T <: Union{ComplexF32, ComplexF64}
    if tune
        coalesced_memory, aim_threads = _get_tuned_params(x, region, forward, inplace)
    else
        coalesced_memory, aim_threads = (0, 0)
    end
    return _make_plan(x, region, forward, inplace; coalesced_memory=coalesced_memory, aim_threads=aim_threads)
end

Base.size(plan::VkFFTPlan) = tuple(plan.buffer_size...)
Base.eltype(plan::VkFFTPlan) = typeof(plan).parameters[1]
AbstractFFTs.plan_fft(x::CuArray{T}, region; tune::Bool=false) where T <: Union{ComplexF32, ComplexF64} = _make_tuned_plan(x, region, true, false; tune=tune)
AbstractFFTs.plan_fft!(x::CuArray{T}, region; tune::Bool=false) where T <: Union{ComplexF32, ComplexF64} = _make_tuned_plan(x, region, true, true; tune=tune)
AbstractFFTs.plan_bfft(x::CuArray{T}, region; tune::Bool=false) where T <: Union{ComplexF32, ComplexF64} = _make_tuned_plan(x, region, false, false; tune=tune)
AbstractFFTs.plan_bfft!(x::CuArray{T}, region; tune::Bool=false) where T <: Union{ComplexF32, ComplexF64} = _make_tuned_plan(x, region, false, true; tune=tune)

function AbstractFFTs.plan_inv(plan::VkFFTPlan)
    # Construct a fake x from plan, and pass that to _make_plan
    dims = plan.dims
    region = plan.region
    x = CuArray{eltype(plan)}(undef, tuple(dims...))
    forward = plan.direction == Int32(-1)

    invplan = _make_plan(x, region, !forward, plan.is_inplace)
    return AbstractFFTs.ScaledPlan(invplan, AbstractFFTs.normalization(x, region))
end

function LinearAlgebra.mul!(y::CuArray{T}, plan::VkFFTPlan, x::CuArray{T}) where T <: Union{ComplexF32, ComplexF64}
    # res = _fft(plan.app, x.storage.buffer.ptr, y.storage.buffer.ptr, plan.direction)
    res = _fft(plan.app, x.data.rc.obj.ptr, y.data.rc.obj.ptr, plan.direction)
    if res != 0
        err = _vkffterr2string(res)
        throw(ArgumentError("Fatal VkFFT error: $err (code: $res)"))
    end

    return y
end

function Base.:*(plan::VkFFTPlan, x::CuArray{T}) where T <: Union{ComplexF32, ComplexF64}
    if plan.is_inplace
        return mul!(x, plan, x)
    else
        y = similar(x)
        return mul!(y, plan, x)
    end
end

end # module
