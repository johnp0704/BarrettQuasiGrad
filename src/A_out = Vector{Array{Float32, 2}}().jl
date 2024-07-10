A_out = Vector{Array{Float32, 2}}()
x_out = Vector{Array{Float32, 2}}()
y_out = Vector{Array{Float32, 2}}()

for i in 1:1000
    A_i = randn(Float32,10,10)
    push!(A_out, A_i)
    
    #key_x = "x_$i"
    x_i = randn(Float32,10,1)
    #x_dict[key_x] = x_gpu
    push!(x_out, x_i)

    #key_y = "y_$i"
    y_i = A_i * x_i
    #y_dict[key_y] = y_gpu
    push!(y_out, y_i)

    println(A_out[i])
    println(x_out[i])
    println(y_out[i])
end

