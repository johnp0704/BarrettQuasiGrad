using CUDA
A_out = Vector{CuArray{Float32, 2}}()
x_out = Vector{CuArray{Float32, 2}}()
y_out = Vector{CuArray{Float32, 2}}()

for i in 1:1000
    A_i = randn(Float32,10,10)
    A_gpu = cu(A_i)
    push!(A_out, A_gpu)
    
    x_i = randn(Float32,10,1)
    x_gpu = cu(x_i)
    push!(x_out, x_gpu)

    y_gpu = A_gpu * x_gpu
    push!(y_out, y_gpu)

    println(A_out[i])
    println(x_out[i])
    println(y_out[i])
end


println()
#this has been added to test pushing


