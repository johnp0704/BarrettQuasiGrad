using CUDA
using BenchmarkTools
using LinearAlgebra

#%% Get device properties
device = CUDA.device()

function get_device_properties(device)
    max_threads_per_block = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_grid_dim_x = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    max_grid_dim_y = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    max_grid_dim_z = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    num_sms = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return Int(max_threads_per_block), Int(max_grid_dim_x), Int(max_grid_dim_y), Int(max_grid_dim_z), Int(num_sms)
end

max_threads_per_block, max_grid_dim_x, max_grid_dim_y, max_grid_dim_z, num_sms = get_device_properties(device)


#%% Matrix-vector kernel
function mat_vec_prod!(y::CuDeviceMatrix{Float64}, A::CuDeviceMatrix{Float64}, x::CuDeviceMatrix{Float64}, offset::Int64)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y + offset

    if i <= size(A, 1) && j <= size(x, 2)
        sum = 0.0 # Use Float64 for consistency with input types
        for h in 1:size(A, 2)
            sum += A[i, h] * x[h, j]
        end
        y[i, j] = sum
    end
    return
end

# Matrix-vector product function
function mat_vec_kernel!(y::CuArray{Float64}, matrices::Vector{CuArray{Float64, 2}}, x::CuArray{Float64}, max_threads_per_block::Int64, max_grid_dim_y::Int64)
    num_matrices = length(matrices)
    combined_y = CUDA.fill(0.0, size(matrices[1], 1), size(x, 2))

    for k in 1:num_matrices
        A = matrices[k]

        # Launch configuration
        threads_x = min(size(A, 1), max_threads_per_block)
        threads_y = min(size(x, 2), div(max_threads_per_block, threads_x))
        blocks_x = cld(size(A, 1), threads_x)
        max_calcs_per_launch = max_grid_dim_y * threads_y

        for offset in 0:max_calcs_per_launch:size(x, 2)-1
            remaining_calcs = min(max_calcs_per_launch, size(x, 2) - offset)
            blocks_y = cld(remaining_calcs, threads_y)
            CUDA.@sync begin
                @cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) mat_vec_prod!(combined_y, A, x, offset)
            end
        end

        # Accumulate the results from each matrix-vector multiplication
        y .+= combined_y
        CUDA.fill!(combined_y, 0.0)  # Reset combined_y for the next matrix
    end
end

#%% How many matrices?
function create_random_cuarrays(num_matrices::Int, rows::Int, cols::Int)
    matrices = Vector{CuArray{Float64, 2}}(undef, num_matrices)
    for i in 1:num_matrices
        matrices[i] = CUDA.fill(rand(), rows, cols)
    end
    return matrices
end

# Matrices setup
num_matrices = 10
rows = 1000
cols = 1000
random_cuarrays = create_random_cuarrays(num_matrices, rows, cols)

#%% Number of calculations to perform
num_calcs = 10000

x = CUDA.fill(rand(), 1000, num_calcs)  # Create a 1000xnum_calcs CuArray with rand values
y = CUDA.fill(0.0, 1000, num_calcs)  # Initialize y as a 1000xnum_calcs CuArray with zeros

#%% Tests timing
@btime CUDA.@sync mat_vec_kernel!(y, random_cuarrays, x, max_threads_per_block, max_grid_dim_y) # Benchmark the function
CUDA.@profile mat_vec_kernel!(y, random_cuarrays, x, max_threads_per_block, max_grid_dim_y)


