"""
    _vkffterr2string(err::UInt64)

Converts a VkFFT error code to a string.

# Arguments
- `err::UInt64`: The error code

# Returns
- A string representation of the error code
"""
function _vkffterr2string(err::UInt64)
    if err == 0
        return "VKFFT_SUCCESS"
    elseif err == 1
        return "VKFFT_ERROR_MALLOC_FAILED"
    elseif err == 2
        return "VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER"
    elseif err == 3
        return "VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER"
    elseif err == 4
        return "VKFFT_ERROR_PLAN_NOT_INITIALIZED"
    elseif err == 5
        return "VKFFT_ERROR_NULL_TEMP_PASSED"
    elseif err == 6
        return "VKFFT_ERROR_MATH_FAILED"
    elseif err == 7
        return "VKFFT_ERROR_FFTdim_GT_MAX_FFT_DIMENSIONS"
    elseif err == 8
        return "VKFFT_ERROR_NONZERO_APP_INITIALIZATION"
    elseif err == 1001
        return "VKFFT_ERROR_INVALID_PHYSICAL_DEVICE"
    elseif err == 1002
        return "VKFFT_ERROR_INVALID_DEVICE"
    elseif err == 1003
        return "VKFFT_ERROR_INVALID_QUEUE"
    elseif err == 1004
        return "VKFFT_ERROR_INVALID_COMMAND_POOL"
    elseif err == 1005
        return "VKFFT_ERROR_INVALID_FENCE"
    elseif err == 1006
        return "VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED"
    elseif err == 1007
        return "VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED"
    elseif err == 1008
        return "VKFFT_ERROR_INVALID_CONTEXT"
    elseif err == 1009
        return "VKFFT_ERROR_INVALID_PLATFORM"
    elseif err == 1010
        return "VKFFT_ERROR_ENABLED_saveApplicationToString"
    elseif err == 1011
        return "VKFFT_ERROR_EMPTY_FILE"
    elseif err == 2001
        return "VKFFT_ERROR_EMPTY_FFTdim"
    elseif err == 2002
        return "VKFFT_ERROR_EMPTY_size"
    elseif err == 2003
        return "VKFFT_ERROR_EMPTY_bufferSize"
    elseif err == 2004
        return "VKFFT_ERROR_EMPTY_buffer"
    elseif err == 2005
        return "VKFFT_ERROR_EMPTY_tempBufferSize"
    elseif err == 2006
        return "VKFFT_ERROR_EMPTY_tempBuffer"
    elseif err == 2007
        return "VKFFT_ERROR_EMPTY_inputBufferSize"
    elseif err == 2008
        return "VKFFT_ERROR_EMPTY_inputBuffer"
    elseif err == 2009
        return "VKFFT_ERROR_EMPTY_outputBufferSize"
    elseif err == 2010
        return "VKFFT_ERROR_EMPTY_outputBuffer"
    elseif err == 2011
        return "VKFFT_ERROR_EMPTY_kernelSize"
    elseif err == 2012
        return "VKFFT_ERROR_EMPTY_kernel"
    elseif err == 2013
        return "VKFFT_ERROR_EMPTY_applicationString"
    elseif err == 2014
        return "VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays"
    elseif err == 2015
        return "VKFFT_ERROR_EMPTY_app"
    elseif err == 3001
        return "VKFFT_ERROR_UNSUPPORTED_RADIX"
    elseif err == 3002
        return "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH"
    elseif err == 3003
        return "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C"
    elseif err == 3004
        return "VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2R"
    elseif err == 3005
        return "VKFFT_ERROR_UNSUPPORTED_FFT_OMIT"
    elseif err == 4001
        return "VKFFT_ERROR_FAILED_TO_ALLOCATE"
    elseif err == 4002
        return "VKFFT_ERROR_FAILED_TO_MAP_MEMORY"
    elseif err == 4003
        return "VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS"
    elseif err == 4004
        return "VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER"
    elseif err == 4005
        return "VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER"
    elseif err == 4006
        return "VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE"
    elseif err == 4007
        return "VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES"
    elseif err == 4008
        return "VKFFT_ERROR_FAILED_TO_RESET_FENCES"
    elseif err == 4009
        return "VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL"
    elseif err == 4010
        return "VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT"
    elseif err == 4011
        return "VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS"
    elseif err == 4012
        return "VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT"
    elseif err == 4013
        return "VKFFT_ERROR_FAILED_SHADER_PREPROCESS"
    elseif err == 4014
        return "VKFFT_ERROR_FAILED_SHADER_PARSE"
    elseif err == 4015
        return "VKFFT_ERROR_FAILED_SHADER_LINK"
    elseif err == 4016
        return "VKFFT_ERROR_FAILED_SPIRV_GENERATE"
    elseif err == 4017
        return "VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE"
    elseif err == 4018
        return "VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE"
    elseif err == 4019
        return "VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER"
    elseif err == 4020
        return "VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE"
    elseif err == 4021
        return "VKFFT_ERROR_FAILED_TO_CREATE_DEVICE"
    elseif err == 4022
        return "VKFFT_ERROR_FAILED_TO_CREATE_FENCE"
    elseif err == 4023
        return "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL"
    elseif err == 4024
        return "VKFFT_ERROR_FAILED_TO_CREATE_BUFFER"
    elseif err == 4025
        return "VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY"
    elseif err == 4026
        return "VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY"
    elseif err == 4027
        return "VKFFT_ERROR_FAILED_TO_FIND_MEMORY"
    elseif err == 4028
        return "VKFFT_ERROR_FAILED_TO_SYNCHRONIZE"
    elseif err == 4029
        return "VKFFT_ERROR_FAILED_TO_COPY"
    elseif err == 4030
        return "VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM"
    elseif err == 4031
        return "VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM"
    elseif err == 4032
        return "VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE"
    elseif err == 4033
        return "VKFFT_ERROR_FAILED_TO_GET_CODE"
    elseif err == 4034
        return "VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM"
    elseif err == 4035
        return "VKFFT_ERROR_FAILED_TO_LOAD_MODULE"
    elseif err == 4036
        return "VKFFT_ERROR_FAILED_TO_GET_FUNCTION"
    elseif err == 4037
        return "VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY"
    elseif err == 4038
        return "VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL"
    elseif err == 4039
        return "VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL"
    elseif err == 4040
        return "VKFFT_ERROR_FAILED_TO_EVENT_RECORD"
    elseif err == 4041
        return "VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION"
    elseif err == 4042
        return "VKFFT_ERROR_FAILED_TO_INITIALIZE"
    elseif err == 4043
        return "VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID"
    elseif err == 4044
        return "VKFFT_ERROR_FAILED_TO_GET_DEVICE"
    elseif err == 4045
        return "VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT"
    elseif err == 4046
        return "VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE"
    elseif err == 4047
        return "VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG"
    elseif err == 4048
        return "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE"
    elseif err == 4049
        return "VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE"
    elseif err == 4050
        return "VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES"
    elseif err == 4051
        return "VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE"
    elseif err == 4052
        return "VKFFT_ERROR_FAILED_TO_CREATE_EVENT"
    elseif err == 4053
        return "VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_LIST"
    elseif err == 4054
        return "VKFFT_ERROR_FAILED_TO_DESTROY_COMMAND_LIST"
    elseif err == 4055
        return "VKFFT_ERROR_FAILED_TO_SUBMIT_BARRIER"
    else
        return "Unknown error"
    end
end

"""
    _new_config(fft_dim::Int64, buffer_dim::Int64, dims::AbstractArray{Int64}, omit_dims::AbstractArray{Int64}, num_batches::Int64, coalesced_memory::Int64, aim_threads::Int64, num_shared_banks::Int64, forward::Bool, double_precision::Bool, inplace::Bool)

Wrapper for the VkFFT C function new_config.
"""
function _new_config(fft_dim::Int64, buffer_dim::Int64, dims::AbstractArray{Int64}, omit_dims::AbstractArray{Int64}, num_batches::Int64, coalesced_memory::Int64, aim_threads::Int64, num_shared_banks::Int64, forward::Bool, double_precision::Bool, inplace::Bool)
    return @ccall libvkfft.new_config(fft_dim::Culonglong, buffer_dim::Int64, dims::Ptr{Culonglong}, omit_dims::Ptr{Culonglong}, num_batches::Culonglong, coalesced_memory::Culonglong, aim_threads::Culonglong, num_shared_banks::Culonglong, forward::Cuchar, double_precision::Cuchar, inplace::Cuchar)::Ptr{Cvoid}
end

"""
    _new_app(config::Ptr{Cvoid}, res_ptr::Ref{Culonglong})

Wrapper for the VkFFT C function new_app.
"""
function _new_app(config::Ptr{Cvoid}, res_ptr::Ref{Culonglong})
    return @ccall libvkfft.new_app(config::Ptr{Cvoid}, res_ptr::Ref{Culonglong})::Ptr{Cvoid}
end

"""
    _fft(app::Ptr{Cvoid}, x::CuPtr{Cvoid}, y::CuPtr{Cvoid}, direction::Int32)

Wrapper for the VkFFT C function fft.
"""
function _fft(app::Ptr{Cvoid}, x::CuPtr{Cvoid}, y::CuPtr{Cvoid}, direction::Int32)
    return @ccall libvkfft.fft(app::Ptr{Cvoid}, x::CuPtr{Cvoid}, y::CuPtr{Cvoid}, direction::Cint)::Cuint
end
