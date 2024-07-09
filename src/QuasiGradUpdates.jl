module QuasiGradUpdates
using CUDA


A_dict = Dict{String, CuArray{Float32, 2}}()
x_dict = Dict{String, CuArray{Float32, 2}}()
y_dict = Dict{String, CuArray{Float32, 2}}()

for i in 1:1000
    key_A = "A_$i"
    A_i = randn(Float32,10,10)
    A_gpu = cu(A_i)
    A_dict[key_A] = A_gpu
    
    key_x = "x_$i"
    x_i = randn(Float32,10,1)
    x_gpu = cu(x_i)
    x_dict[key_x] = x_gpu

    key_y = "y_$i"
    y_gpu = A_gpu * x_gpu
    y_dict[key_y] = y_gpu
end
println(x_dict["x_156"])
println(y_dict["y_290"])

end