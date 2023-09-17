#include <iostream>
#include <fstream>
#include "Header.h"

RobotDimensions dim_of_SCARE() {
    struct RobotDimensions scare{};

    scare.pI = 3.141592653589793238462643383279502884197169f;
    scare.body = 185.0f;
    scare.coxa_angle_deg = 60.0f;
    scare.coxa_length = 165.0f;
    scare.tibia_angle_deg = 120.0f; //90
    scare.tibia_length = 190.0f;
    scare.tibia_length_squared = scare.tibia_length * scare.tibia_length;
    scare.femur_angle_deg = 150.0f; //120
    scare.femur_length = 300.0f; //200
    scare.max_angle_coxa = scare.pI / 180.0f * scare.coxa_angle_deg;
    scare.min_angle_coxa = -scare.pI / 180.0f * scare.coxa_angle_deg;
    scare.max_angle_coxa_w_margin = scare.pI / 180.0f * (scare.coxa_angle_deg - 10.0f);
    scare.min_angle_coxa_w_margin = -scare.pI / 180.0f * (scare.coxa_angle_deg - 10.0f);
    scare.max_angle_tibia = scare.pI / 180.0f * scare.tibia_angle_deg;
    scare.min_angle_tibia = -scare.pI / 180.0f * scare.tibia_angle_deg;
    scare.max_angle_femur = scare.max_angle_tibia;
    scare.min_angle_femur = scare.min_angle_tibia;
    scare.max_angle_femur_w_margin = scare.pI / 180.0f * (scare.tibia_angle_deg + 20.0f);
    scare.min_angle_femur_w_margin = -scare.pI / 180.0f * (scare.tibia_angle_deg + 20.0f);
    scare.max_tibia_to_gripper_dist = scare.tibia_length + scare.femur_length;

    scare.positiv_saturated_femur[0] = cos(scare.max_angle_femur) * scare.femur_length;
    scare.positiv_saturated_femur[1] = sin(scare.max_angle_femur) * scare.femur_length;

    scare.negativ_saturated_femur[0] = cos(scare.min_angle_femur) * scare.femur_length;
    scare.negativ_saturated_femur[1] = sin(scare.min_angle_femur) * scare.femur_length;

    scare.fem_tib_min_host[0] = scare.tibia_length + scare.femur_length * cos(scare.pI / 180.0f * scare.femur_angle_deg);
    scare.fem_tib_min_host[1] = scare.femur_length * sin(scare.pI / 180.0f * scare.femur_angle_deg);

    scare.min_tibia_to_gripper_dist = sqrt(scare.fem_tib_min_host[0] * scare.fem_tib_min_host[0]
                                           + scare.fem_tib_min_host[1] * scare.fem_tib_min_host[1]);
    scare.middle_TG = (scare.max_tibia_to_gripper_dist + scare.min_tibia_to_gripper_dist) / 2.0f;

    return scare;
}

__global__
void empty_kernel() {
}

// Function to calculate the mean of an array of floats
float calculateMean(const float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum / size;
}

// Function to calculate the standard deviation of an array of floats
float calculateStdDev(const float* arr, int size, float mean) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = arr[i] - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / (size - 1));
}

__device__ float sumOfSquares3df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
}

__device__
void dist_noflip(float* point, RobotDimensions& dim, float* result_point)
// no angle flipping
{
    // Coxa as the frame of reference without rotation
    float result[3];
    result[0] = point[0] - dim.body;
    result[1] = point[1];
    result[2] = point[2];

    // finding coxa angle
    float required_angle_coxa = atan2f(result[1], result[0]);

    // flipping angle if above +-90deg
    // required_angle_coxa = fmodf(required_angle_coxa + dim.pI / 2.f + 2.f * dim.pI, dim.pI) - dim.pI / 2.f;

    // saturating coxa angle for dist
    required_angle_coxa = fmaxf(fminf(required_angle_coxa, dim.max_angle_coxa_w_margin), dim.min_angle_coxa_w_margin);

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox = cosf(-required_angle_coxa);
    float sin_angle_cox = sinf(-required_angle_coxa);
    float buffer = result[0] * sin_angle_cox;
    result[0] = result[0] * cos_angle_cox - result[1] * sin_angle_cox;
    result[1] = buffer + result[1] * cos_angle_cox;

    // Femur as the frame of reference witout rotation
    result[0] -= dim.coxa_length;

    // finding femur angle
    float required_angle_femur = atan2f(result[2], result[0]);

    // saturating coxa angle for dist
    required_angle_femur = fmaxf(fminf(required_angle_femur, dim.max_angle_femur_w_margin), dim.min_angle_femur_w_margin);

    // canceling femur rotation for dist
    float cos_angle_fem = cosf(required_angle_femur);
    float sin_angle_fem = sinf(required_angle_femur);

    // middle_TG as the frame of reference
    result[0] -= dim.middle_TG * cos_angle_fem;
    result[2] -= dim.middle_TG * sin_angle_fem;

    // rotating back to default xyz, but staying on middle_TG

    buffer = result[1] * sin_angle_cox;
    result[1] = -result[0] * sin_angle_cox + result[1] * cos_angle_cox;
    result[0] = result[0] * cos_angle_cox + buffer;

    result_point[0] = result[0];
    result_point[1] = result[1];
    result_point[2] = result[2];

    return;
}

__device__
void dist_flip(float* point, RobotDimensions& dim, float* result_point)
// with angle flipping
{
    // Coxa as the frame of reference without rotation
    float result[3];
    result[0] = point[0] - dim.body;
    result[1] = point[1];
    result[2] = point[2];

    // finding coxa angle
    float required_angle_coxa = atan2f(-result[1], -result[0]);

    // flipping angle if above +-90deg
    // required_angle_coxa = fmodf(required_angle_coxa + dim.pI / 2.f + 2.f * dim.pI, dim.pI) - dim.pI / 2.f;

    // saturating coxa angle for dist
    required_angle_coxa = fmaxf(fminf(required_angle_coxa, dim.max_angle_coxa_w_margin), dim.min_angle_coxa_w_margin);

    // canceling coxa rotation for dist
    // Coxa as the frame of reference with rotation
    float cos_angle_cox = cosf(-required_angle_coxa);
    float sin_angle_cox = sinf(-required_angle_coxa);
    float buffer = result[0] * sin_angle_cox;
    result[0] = result[0] * cos_angle_cox - result[1] * sin_angle_cox;
    result[1] = buffer + result[1] * cos_angle_cox;

    // Femur as the frame of reference witout rotation
    result[0] -= dim.coxa_length;

    // finding femur angle
    float required_angle_femur = atan2f(result[2], result[0]);

    // saturating coxa angle for dist
    required_angle_femur = fmaxf(fminf(required_angle_femur, dim.max_angle_femur_w_margin), dim.min_angle_femur_w_margin);

    // canceling femur rotation for dist
    float cos_angle_fem = cosf(required_angle_femur);
    float sin_angle_fem = sinf(required_angle_femur);

    // middle_TG as the frame of reference
    result[0] -= dim.middle_TG * cos_angle_fem;
    result[2] -= dim.middle_TG * sin_angle_fem;

    // rotating back to default xyz, but staying on middle_TG

    buffer = result[1] * sin_angle_cox;
    result[1] = -result[0] * sin_angle_cox + result[1] * cos_angle_cox;
    result[0] = result[0] * cos_angle_cox + buffer;

    result_point[0] = result[0];
    result_point[1] = result[1];
    result_point[2] = result[2];

    return;
}

__device__
void dist_double_sol(float* point, RobotDimensions& dim, float* result_point)
//
{
    // Coxa as the frame of reference without rotation
    float result_noflip[3];
    float result_flip[3];

    dist_noflip(point, dim, result_noflip);
    dist_flip(point, dim, result_flip);

    float* result_to_use = (sumOfSquares3df(result_noflip) < sumOfSquares3df(result_flip)) ? result_noflip : result_flip;
    // result_to_use = result_flip;
    result_point[0] = result_to_use[0];
    result_point[1] = result_to_use[1];
    result_point[2] = result_to_use[2];

}

// Kernel launch function
__global__
void dist_kernel(Matrix table, RobotDimensions dimensions, Matrix result_table)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        dist_double_sol((table.elements + i * table.width), dimensions, (result_table.elements + i * result_table.width));
    }
}

__global__
void norm3df_kernel(Matrix table, Matrix result_table)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        result_table.elements[i] = norm3df(table.elements[i * table.width],
                                           table.elements[i * table.width+1],
                                           table.elements[i * table.width+2]);
    }
}

__global__
void change_z_kernel(Matrix table, float zval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        table.elements[i * table.width+2] = zval;
    }
}

AutoEstimator::AutoEstimator(int pxWidth, int pxHeight) {
    blockSize = 1024;
    verbose = true;
    screenWidth = pxWidth;
    screenHeight = pxHeight;
    dimensions = dim_of_SCARE();
    rows = screenWidth * screenHeight;

    table_input.width = 3; table_input.height = rows;
    table_input.elements = new float[table_input.width * table_input.height];

    result.width = 3; result.height = rows;
    result.elements = new float[table_input.width * table_input.height];

    result_norm.width = 1; result_norm.height = rows;
    result_norm.elements = new float[table_input.width * table_input.height];

    table_input_gpu.width = table_input.width; table_input_gpu.height = table_input.height;
    result_gpu.width = table_input_gpu.width; result_gpu.height = table_input_gpu.height;
    result_norm_gpu.width = result_norm.width; result_norm_gpu.height = table_input_gpu.height;

    numBlocks = (rows + blockSize - 1) / blockSize;
    error_check();
    setup_kernel();
    input_as_grid();
    alocate_gpu_mem();
    copy_input_cpu2gpu();
}

void AutoEstimator::error_check(){
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or exit the program as needed.
    }
}

void AutoEstimator::input_as_grid(){
    for (int i = 0; i < screenHeight; i++) {
        for (int j = 0; j < screenWidth; j++) {
            int row = i * screenWidth + j;
            // X
            *(table_input.elements + row * table_input.width + 0)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = (float)(j - screenWidth /2);
            // Y
            *(table_input.elements + row * table_input.width + 1)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = -(float)(i - screenHeight / 2);
            // Z
            *(table_input.elements + row * table_input.width + 2)
                    //= -500.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f;
                    = 0.f;
        }
    }
    if (verbose) { std::cout << "Host grid generated" << std::endl; }
}

void AutoEstimator::change_zvalue(float zvalue){

    change_z_kernel<<<numBlocks, blockSize >>>(table_input_gpu, zvalue);

}

void AutoEstimator::alocate_gpu_mem(){
    cudaMalloc(&table_input_gpu.elements, table_input_gpu.width * table_input_gpu.height * sizeof(float));

    cudaMalloc(&result_gpu.elements, result_gpu.width * result_gpu.height * sizeof(float));

    cudaMalloc(&result_norm_gpu.elements, result_norm_gpu.width * result_norm_gpu.height * sizeof(float));

    if (verbose) { std::cout << "GPU memory allocated" << std::endl; }
    error_check();
}

void AutoEstimator::copy_input_cpu2gpu(){
    cudaMemcpy(table_input_gpu.elements,
               table_input.elements,
               table_input.width * table_input.height * sizeof(float),
               cudaMemcpyHostToDevice);
    if (verbose) { std::cout << "Host data copied to GPU" << std::endl; }
    error_check();
}

void AutoEstimator::setup_kernel(){
    std::wcout << "Threads per block: " << blockSize << "\nNumber of blocks: " << numBlocks << std::endl;
    empty_kernel<<<numBlocks, blockSize >>>();
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Empty kernels started" << std::endl;}
    error_check();
}

void AutoEstimator::compute_dist(){
    if (verbose) { std::cout << "Compute started" << std::endl;}
    dist_kernel<<<numBlocks, blockSize >>>(table_input_gpu, dimensions, result_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::compute_result_norm(){
    if (verbose) { std::cout << "Compute norm started" << std::endl;}
    norm3df_kernel<<<numBlocks, blockSize >>>(result_gpu, result_norm_gpu);
    cudaDeviceSynchronize();
    if (verbose) { std::cout << "Compute done" << std::endl;}
    error_check();
}

void AutoEstimator::copy_output_gpu2cpu(){
    cudaMemcpy(result.elements,
               result_gpu.elements,
               result_gpu.width * result_gpu.height * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(result_norm.elements,
               result_norm_gpu.elements,
               result_norm_gpu.width * result_norm_gpu.height * sizeof(float),
               cudaMemcpyDeviceToHost);
    if (verbose) { std::cout << "GPU result copied to host" << std::endl;}
    cudaDeviceSynchronize();
    error_check();
}

void AutoEstimator::delete_all(){
    delete[] table_input.elements;
    cudaFree(table_input_gpu.elements);
    cudaFree(result_gpu.elements);
    cudaFree(result_norm_gpu.elements);
    delete[] result.elements;
    delete[] result_norm.elements;
    if (verbose) { std::cout << "All pointers deleted" << std::endl;}
    error_check();
}

int main_le_old(void)
{
    AutoEstimator autoe{6, 7};
    std::cout << "initializing c++\n";
    //getting scare's dimensions in an object
    const RobotDimensions dimensions = dim_of_SCARE();

    // number of rows is the size of a 2K screen * 10
    int screenWidth = 1920;
    int screenHeight = 1080;
    int scale_up_factor = 1;
    int rows = screenWidth * screenHeight * scale_up_factor; // 1 << 20 is equivalent to 2,097,152.

    // Creates the matrix object (not a matrix but row/col are stored in row major) so:
    // M(row, col) = *(table_input.elements + row * table_input.width + col)
    Matrix table_input;
    table_input.width = 3; table_input.height = rows;
    table_input.elements = new float[table_input.width * table_input.height];

    // Initialize the vectorList with random values within -1000 1000
    for (int i = 0; i < screenHeight; i++) {
        for (int j = 0; j < screenWidth; j++) {
            int row = i * screenWidth + j;
            // X
            *(table_input.elements + row * table_input.width + 0)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = (float)(j - screenWidth /2);
            // Y
            *(table_input.elements + row * table_input.width + 1)
                    //= -1000.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2000.0f;
                    = -(float)(i - screenHeight / 2);
            // Z
            *(table_input.elements + row * table_input.width + 2)
                    //= -500.0f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 1000.0f;
                    = 200.f;
        }
    }
    // *(table_input.elements + 0 * table_input.width + 0) = 600.f;
    // *(table_input.elements + 0 * table_input.width + 1) = 0.f;
    // *(table_input.elements + 0 * table_input.width + 2) = 0.f;
    //
    // *(table_input.elements + 1 * table_input.width + 0) = 700.f;
    // *(table_input.elements + 1 * table_input.width + 1) = 0.f;
    // *(table_input.elements + 1 * table_input.width + 2) = 0.f;

    std::cout << "allocating GPU memory\n";

    // Allocating memory on the gpu for the input
    Matrix table_input_gpu;
    table_input_gpu.width = table_input.width; table_input_gpu.height = table_input.height;
    cudaMalloc(&table_input_gpu.elements, table_input_gpu.width * table_input_gpu.height * sizeof(float));

    // and the output
    Matrix result_gpu;
    result_gpu.width = table_input_gpu.width; result_gpu.height = table_input_gpu.height;
    cudaMalloc(&result_gpu.elements, result_gpu.width * result_gpu.height * sizeof(float));

    // Copying from  host to gpu (only for the input)
    cudaMemcpy(table_input_gpu.elements, table_input.elements, table_input.width * table_input.height * sizeof(float), cudaMemcpyHostToDevice);


    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run kernel on 1M elements on the GPU
    int blockSize = 1024; // threads per block (multiples of 32 are best)
    int numBlocks = (rows + blockSize - 1) / blockSize; // number of blocks
    std::wcout << "Threads per block: " << blockSize << "\nNumber of blocks: " << numBlocks << std::endl;
    // int blockSize = rows; // threads per block
    // int numBlocks = 1; // number of blocks

    // Add an "empty" kernel to incur setup overhead
    empty_kernel<<<numBlocks, blockSize >>>();

    std::cout << "Running batch of kernel\n";
    // Calculate execution time per row
    // for several runs
    int const runs = 50;
    float milliseconds;
    float executionTimePerRow;
    float time_vect[runs];
    float time_vect_per_row[runs];
    cudaError_t cudaStatus;

    for (int index = 0; index < runs; ++index) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start time
        cudaEventRecord(start);

        //Compute starts
        dist_kernel<<<numBlocks, blockSize >>>(table_input_gpu, dimensions, result_gpu);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Record the end time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        //extract time from GPU
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        //converts times and stores in the vector
        executionTimePerRow = milliseconds / (float)rows * 1000000.f;
        time_vect[index] = milliseconds;
        time_vect_per_row[index] = executionTimePerRow;
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or exit the program as needed.
    }

    float mean = calculateMean(time_vect, runs);

    std::cout << "\nTotal compute time for " << rows/1'000'000 << " M rows is " << mean << " ms on avg \nThat's " << 1.f / mean * 1e3 * (float)scale_up_factor << " Hz on a 2k screen" << "\n";

    // Calculate the average
    mean = calculateMean(time_vect_per_row, runs);
    float stddev = calculateStdDev(time_vect_per_row, runs, mean);

    // Calculate the standard deviation
    std::cout << "Execution time per row: " << mean << "ns\n(std.dev.of " << stddev << ")" << std::endl;

    // extract GPU memory onto host
    // Allocate
    float* result_host = new float[result_gpu.width * result_gpu.height];
    // Copy data from GPU to host
    cudaMemcpy(result_host, result_gpu.elements, result_gpu.width* result_gpu.height * sizeof(float), cudaMemcpyDeviceToHost);

    // ################################ WRITING TO BIN vvvvvvv
    // Specify the file path
    const char* binaryFilePath = "result.bin";

    // Open the binary file for writing
    std::ofstream outputFile(binaryFilePath, std::ios::binary);

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the binary file for writing." << std::endl;
    }
    else {
        // Write the data to the binary file
        outputFile.write(reinterpret_cast<const char*>(result_host), sizeof(float) * result_gpu.width * result_gpu.height);

        // Close the binary file
        outputFile.close();
        std::cout << "Data has been written to " << binaryFilePath << std::endl;
    }
    // ################################ WRITING TO BIN ^^^^

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(table_input_gpu.elements);
    cudaFree(result_gpu.elements);
    delete[] table_input.elements;
    delete[] result_host;

    return 0;
}