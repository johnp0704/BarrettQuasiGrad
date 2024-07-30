using CUDA

# Gets CUDA device
device = CUDA.device()

#Device Properties Function
function print_device_properties(device)
    name = CUDA.name(device)
    max_threads_per_block = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_grid_dim_x = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
    max_grid_dim_y = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
    max_grid_dim_z = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)
    max_threads_per_multiprocessor = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    num_sms = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    # Print properties
    println("Device: $name")
    println("Max threads per block: $max_threads_per_block")
    println("Max grid size (X): $max_grid_dim_x")
    println("Max grid size (Y): $max_grid_dim_y")
    println("Max grid size (Z): $max_grid_dim_z")
    println("Max threads per multiprocessor: $max_threads_per_multiprocessor")
    println("Number of SMs: $num_sms")
end

#Call Function
print_device_properties(device)









