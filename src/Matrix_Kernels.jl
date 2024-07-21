using CUDA
using BenchmarkTools
using LinearAlgebra

#%% Sets kernel
function mat_vec_kernel!(y::CuArray{Float64, 1, CUDA.DeviceMemory}, A::CuArray{Float64, 2, CUDA.DeviceMemory}, x::CuArray{Float64, 1, CUDA.DeviceMemory})
    i = threadIdx().x + (blockIdx().x -1) * blockDim().x
    if i <= size(A,1)
        sum = 0.0 #sets sum using Float64
        for h in 1:size(A,2)
            sum += A[i, h] * x[h]
        end
        y[i] = sum
    end
    return
end

#%%Sets product function
function mat_vec_prod!(y::CuArray{Float64, 1, CUDA.DeviceMemory}, A::CuArray{Float64, 2, CUDA.DeviceMemory}, x::CuArray{Float64, 1, CUDA.DeviceMemory})
    kernel = @cuda launch=false mat_vec_kernel!(y::CuArray{Float64, 1, CUDA.DeviceMemory}, A::CuArray{Float64, 2, CUDA.DeviceMemory}, x::CuArray{Float64, 1, CUDA.DeviceMemory})
    config = launch_configuration(kernel.fun)
    threads = min(size(A, 1), config.threads)
    blocks = cld(size(A, 1), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks mat_vec_kernel!(y::CuArray{Float64, 1, CUDA.DeviceMemory}, A::CuArray{Float64, 2, CUDA.DeviceMemory}, x::CuArray{Float64, 1, CUDA.DeviceMemory})
    end
end

#%% Sets matrices
A = CUDA.fill(rand(), 1000,1000)  # Create CuArray of rand values (0-1) 
x = CUDA.fill(rand(), 1000)  
y = similar(x)


#%%Tests timing
@time mat_vec_prod!(y, A, x) #~0.000789 seconds, 65 allocations
@btime mat_vec_prod!(y, A, x)#~21.500 us, 61 allocations



#%% Verifies functionality
y_cpu = Array(y)  #Copy result back to CPU for verification
A_cpu = Array(A)
x_cpu = Array(x)
println(norm(y_cpu - A_cpu * x_cpu) < 1e-5)  #Print true if the multiplication was successful


#try 1000x1000 matrices (or larger) -----
#do many matrix vector products, not just one, multiple parallel (try one kernel that calls a different block for different functions)
#figure out blocks and threads -----
#fix timing operations (get statistics, lab notebook)
#try more tricks, two different operations or such (sin exponential, etc.)
#type stabilize, add typed into function arguments (makes allocations less) -----

