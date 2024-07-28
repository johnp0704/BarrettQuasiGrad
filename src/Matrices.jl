using CUDA
using BenchmarkTools

#%% Initialize common vector x
x_length = 1000
x_gpu = CUDA.fill(rand(Float64), 1000, x_length)

#%% Generate number of random arrays
function create_random_cuarrays(num_matrices::Int, rows::Int, cols::Int)
    matrices = Vector{CuArray{Float64, 2}}(undef, num_matrices)
    for i in 1:num_matrices
        matrices[i] = CUDA.fill(rand(Float64), rows, cols)
    end
    return matrices
end

# Example usage
num_matrices = 5
rows = 1000
cols = 1000
random_cuarrays = create_random_cuarrays(num_matrices, rows, cols)

#%% GPU calculation function
function y_calc_gpu(random_cuarrays::Vector{CuArray{Float64, 2}}, x_gpu::CuArray{Float64, 2}, y_arrays::Vector{CuArray{Float64, 2}})
    for i in 1:length(random_cuarrays)
        push!(y_arrays, random_cuarrays[i] * x_gpu)
    end
end

#%% Initialize y_arrays as an empty Vector of CuArray{Float64, 2}
y_arrays = Vector{CuArray{Float64, 2}}()

# Timing
@time y_calc_gpu(random_cuarrays, x_gpu, y_arrays) #gives ~0.005614 seconds, 711 allocations

#%% btime
y_arrays = Vector{CuArray{Float64, 2}}()  # Reset y_arrays
@btime y_calc_gpu(random_cuarrays, x_gpu, y_arrays) #gives ~24.500 us, 104 allocations

#%% Verify result (optional)
for i in 1:length(y_arrays)
    println("y_array $i size: ", size(y_arrays[i]))
end








#%% CPU

B = randn(10,10)

w = randn(10,1)

function z_calc_cpu(B, w)
    return B * w
end
z = similar(x)
@time z .= z_calc_cpu(B, w) #~0.003634 seconds, 388 allocations
@btime z .= z_calc_cpu(B, w) #~130.000 ns, 2 allocations