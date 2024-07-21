using CUDA
using BenchmarkTools
#A_out = Vector{CuArray{Float32, 2}}()
#x_out = Vector{CuArray{Float32, 2}}()
#y_out = Vector{CuArray{Float32, 2}}()

#for i in 1:1000 #don't loop, all at once!
 #   A_i = randn(Float32,10,10)
  #  A_gpu = cu(A_i)
   # push!(A_out, A_gpu)
    
   # x_i = randn(Float32,10,1)
    #x_gpu = cu(x_i)
    #push!(x_out, x_gpu)

    #y_gpu = A_gpu * x_gpu
    #push!(y_out, y_gpu)

    #println(A_out[i])
    #println(x_out[i])
    #println(y_out[i])
#end
#Above is old solution



#x = randn(1000,10)
#function x_squared(x::Matrix{Float64}, y) 
    #y .= x .^2   #y . says take every element of array and overwrite on y, makes it not allocate memory
#end

#y = similar(x)
#@time x_squared(x,y)

    #@btime #runs like 1000 iterations of the code
#@btime x_squared(x,y) #evaluates 1000 times, then give statistics

#Above is all for CPU only, example code




#%% GPU

A = randn(1000,1000)
A_gpu = cu(A)

x = randn(1000,1)
x_gpu = cu(x)

function y_calc_gpu(A_gpu::CuArray{Float32,2}, x_gpu::CuArray{Float32,2})
    return A_gpu * x_gpu
end
y_gpu = similar(x_gpu)

@time y_gpu .= y_calc_gpu(A_gpu, x_gpu) #gives ~0.005614 seconds, 711 allocations
@btime y_gpu .= y_calc_gpu(A_gpu, x_gpu) #gives ~24.500 us, 104 allocations


#%% CPU

B = randn(10,10)

w = randn(10,1)

function z_calc_cpu(B, w)
    return B * w
end
z = similar(x)
@time z .= z_calc_cpu(B, w) #~0.003634 seconds, 388 allocations
@btime z .= z_calc_cpu(B, w) #~130.000 ns, 2 allocations