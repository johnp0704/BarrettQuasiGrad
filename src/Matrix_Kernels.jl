using CUDA
using BenchmarkTools
using LinearAlgebra

#%% Sets kernel
function mat_vec_kernel!(y, A, x)
    i = threadIdx().x + (blockIdx().x -1) * blockDim().x
    if i <= size(A,1)
        sum = 0.0f0 #sets sum using Float32
        for h in 1:size(A,2)
            sum += A[i, h] * x[h]
        end
        y[i] = sum
    end
    return
end

#%%Sets product function
function mat_vec_prod!(y, A, x)
    kernel = @cuda launch=false mat_vec_kernel!(y, A, x)
    config = launch_configuration(kernel.fun)
    threads = min(size(A, 1), config.threads)
    blocks = cld(size(A, 1), threads)

    CUDA.@sync begin
        @cuda threads=threads blocks=blocks mat_vec_kernel!(y, A, x)
    end
end

#%% Sets matrices
A = CUDA.fill(rand(), 10,10)  # Create a 10x10 CuArray of rand values (0-1) 
x = CUDA.fill(rand(), 10)  # Create a 10x1 CuArray with rand values (0-1)
y = similar(x)


#%%Tests timing
@time mat_vec_prod!(y, A, x) #~0.000789 seconds, 65 allocations
@btime mat_vec_prod!(y, A, x)#~21.500 us, 61 allocations



#%% Verifies functionality
y_cpu = Array(y)  # Copy the result back to the CPU for verification
A_cpu = Array(A)
x_cpu = Array(x)
println(norm(y_cpu - A_cpu * x_cpu) < 1e-5)  # Should print true if the multiplication was successful