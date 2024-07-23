using CUDA
using BenchmarkTools
using LinearAlgebra

#%% Get device properties
device = CUDA.device()

# Function to get device properties
function get_device_properties(device)
    max_threads_per_block = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_grid_dim_x = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    max_grid_dim_y = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    max_grid_dim_z = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    num_sms = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return Int(max_threads_per_block), Int(max_grid_dim_x), Int(max_grid_dim_y), Int(max_grid_dim_z), Int(num_sms)
end

# Get properties for the current device
max_threads_per_block, max_grid_dim_x, max_grid_dim_y, max_grid_dim_z, num_sms = get_device_properties(device)

#%% Matrix-vector kernel
function mat_vec_kernel!(y::CuArray{Float64}, A::CuArray{Float64}, x::CuArray{Float64}, offset::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y + offset
    if i <= size(A, 1) && j <= size(x, 2)
        sum = 0.0 
        for h in 1:size(A, 2)
            sum += A[i, h] * x[h, j]
        end
        y[i, j] = sum
    end
    return
end

# Matrix-vector product function
function mat_vec_prod!(y::CuArray{Float64}, A::CuArray{Float64}, x::CuArray{Float64}, max_threads_per_block::Int, max_grid_dim_y::Int)
    # Launch configuration
    threads_x = min(size(A, 1), max_threads_per_block)
    threads_y = min(size(x, 2), div(max_threads_per_block, threads_x))
    blocks_x = cld(size(A, 1), threads_x)
    max_calcs_per_launch = max_grid_dim_y * threads_y

    for offset in 0:max_calcs_per_launch:size(x, 2)-1
        remaining_calcs = min(max_calcs_per_launch, size(x, 2) - offset)
        blocks_y = cld(remaining_calcs, threads_y)
        CUDA.@sync begin
            @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) mat_vec_kernel!(y, A, x, offset)
        end
    end
end

#%% Number of calculations to perform
num_calcs = 200000

#%% Sets matrices
A = CUDA.fill(rand(), 1000, 1000)  # Create a 1000x1000 CuArray of rand values
x = CUDA.fill(rand(), 1000, num_calcs)  # Create a 1000xnum_calcs CuArray with rand values
y = similar(x)

#%% Tests timing
@time mat_vec_prod!(y, A, x, max_threads_per_block, max_grid_dim_y) # Measure time taken for matrix-vector multiplication
@btime mat_vec_prod!(y, A, x, max_threads_per_block, max_grid_dim_y) # Benchmark the function

#%% Verifies functionality
y_cpu = Array(y)  # Copy result back to CPU for verification
A_cpu = Array(A)
x_cpu = Array(x)
verification = all(norm(y_cpu[:, j] - A_cpu * x_cpu[:, j]) < 1e-5 for j in 1:num_calcs)
println(verification)  # Print true if the multiplication was successful
