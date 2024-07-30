using CUDA
using BenchmarkTools

#%% Generate Common array
x_length = 1000
x_gpu = CUDA.fill(rand(Float64), 1000, x_length)


#%% Generate Random arrays
function create_random_cuarrays(num_matrices::Int, rows::Int, cols::Int)
    matrices = Vector{CuArray{Float64, 2}}(undef, num_matrices)
    for i in 1:num_matrices
        matrices[i] = CUDA.fill(rand(Float64), rows, cols)
    end
    return matrices
end

# Random Array Setup
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

#%% Initialize Result arrays
y_arrays = Vector{CuArray{Float64, 2}}()

# Timing
@time y_calc_gpu(random_cuarrays, x_gpu, y_arrays) 

#%% btime
y_arrays = Vector{CuArray{Float64, 2}}()  # Reset y_arrays
@btime y_calc_gpu(random_cuarrays, x_gpu, y_arrays) 




#%% CPU
x_cpu_length = 1000
x_cpu = fill(rand(Float64), 1000, x_cpu_length)

#%%
function create_random_arrays(num_matrices_cpu::Int, rows_cpu::Int, cols_cpu::Int)
    matrices_cpu = Vector{Array{Float64, 2}}(undef, num_matrices_cpu)
    for i in 1:num_matrices_cpu
        matrices_cpu[i] = fill(rand(Float64), rows_cpu, cols_cpu)
    end
    return matrices_cpu
end

#Set up random matrices
num_matrices_cpu = 5
rows_cpu = 1000
cols_cpu = 1000
random_arrays_cpu = create_random_arrays(num_matrices_cpu, rows_cpu, cols_cpu)

#%% CPU calculation function
function y_calc_cpuzz(A::Vector{Matrix{Float64}}, random_arrays_cpu::Vector{Array{Float64, 2}}, x_cpu::Array{Float64, 2}, y_cpu_arrays::Array{Float64, 2}) #y_cpu_arrays::Vector{Array{Float64, 2}})
    for i in 1:length(random_arrays_cpu)
        #push!(y_cpu_arrays, random_arrays_cpu[i] * x_cpu)
        # for 2d arrays
        mul!(A[i], random_arrays_cpu[i], x_cpu)
        #mul!(y_cpu_arrays[:,(1:1000) .+ (i-1)*1000], random_arrays_cpu[i], x_cpu)
        #y_cpu_arrays[:,(1:1000) .+ (i-1)*1000] .= 
        #random_arrays_cpu[i] * x_cpu

        # for 3d array
        #y_cpu_arrays[i] = random_arrays_cpu[i] * x_cpu
    end
end


#%%Initialize Result arrays
y_cpu_arrays = Vector{Array{Float64, 2}}()
y_cpu_arrays = zeros(rows_cpu, num_matrices_cpu*cols_cpu)

#%% Timing
A = [randn(1000,1000) for ii in 1:5]
y_cpu_arrays = zeros(rows_cpu, num_matrices_cpu*cols_cpu)

@time y_calc_cpuzz(A, random_arrays_cpu, x_cpu, y_cpu_arrays)

#%% btime
y_cpu_arrays = Vector{Array{Float64, 2}}()
@btime y_calc_cpu(random_arrays_cpu, x_cpu, y_cpu_arrays) 

# %% 
function matmat(A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64})
    return mul!(C, A, B)
end

A = randn(500,500)
B = randn(500,500)
C = randn(500,500)

# %%
@btime matmat(A,B,C)