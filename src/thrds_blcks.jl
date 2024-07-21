using CUDA

# Gets active device
device = CUDA.device()

# Function to print device properties
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
    println("Maximum threads per block: $max_threads_per_block")
    println("Maximum grid size (X): $max_grid_dim_x")
    println("Maximum grid size (Y): $max_grid_dim_y")
    println("Maximum grid size (Z): $max_grid_dim_z")
    println("Maximum threads per multiprocessor: $max_threads_per_multiprocessor")
    println("Number of SMs: $num_sms")
end

#Run function
print_device_properties(device)