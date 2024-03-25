#include "cuda_util.cuh"

__global__ void empty_kernel() {}

// __host__ float norm3df(float x, float y, float z) {
//     return sqrtf(x * x + y * y + z * z);
// };

// __device__ float sumOfSquares3df(const float* vector) {
//     return vector[0] * vector[0] + vector[1] * vector[1] +
//            vector[2] * vector[2];
// }
//
// __device__ float sumOfSquares2df(const float* vector) {
//     return vector[0] * vector[0] + vector[1] * vector[1];
// }
//
__global__ void norm3df_kernel(Arrayf3 input, Arrayf output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = norm3df(input.elements[i].x, input.elements[i].y,
                                     input.elements[i].z);
    }
}

// #define CUDA_CHECK_ERROR(errorMessage)                                         \
//     do {                                                                       \
//         cudaError_t err = cudaGetLastError();                                  \
//         if (err != cudaSuccess) {                                              \
//             fprintf(stderr, "CUDA error in %s: %s\n", errorMessage,            \
//                     cudaGetErrorString(err));                                  \
//             exit(EXIT_FAILURE);                                                \
//         }                                                                      \
//     } while (0)

// /**
//  * @brief applies kernel on provided array to provided array
//  *
//  * @tparam T_in
//  * @tparam T_out
//  * @param input Input array of any type
//  * @param dim dimension of the leg
//  * @param kernel cuda kernel to be used
//  * @param output result is stored here
//  */
// template <typename T_in, typename T_out>
// void apply_kernel(const Array<T_in> input, const LegDimensions dim,
//                   void (*kernel)(const Array<T_in>, const LegDimensions,
//                                  Array<T_out> const),
//                   Array<T_out> const output) {
//     Array<T_in> gpu_in{};
//     Array<T_out> gpu_out{};
//     gpu_in.length = input.length;
//     gpu_out.length = output.length;
//     cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(T_in));
//     CUDA_CHECK_ERROR("cudaMalloc gpu_in.elements");
//     cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(T_out));
//     CUDA_CHECK_ERROR("cudaMalloc gpu_out.elements");
//
//     cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(T_in),
//                cudaMemcpyHostToDevice);
//     CUDA_CHECK_ERROR("cudaMemcpy gpu_in.elements");
//
//     int blockSize = 1024;
//     int numBlock = (input.length + blockSize - 1) / blockSize;
//     kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out);
//     cudaDeviceSynchronize();
//     CUDA_CHECK_ERROR("Kernel launch");
//
//     cudaMemcpy(output.elements, gpu_out.elements, output.length * sizeof(T_out),
//                cudaMemcpyDeviceToHost);
//     CUDA_CHECK_ERROR("cudaMemcpy gpu_out.elements");
//     cudaDeviceSynchronize();
//
//     cudaFree(gpu_in.elements);
//     cudaFree(gpu_out.elements);
// }
//
// // Explicit instantiation for float3, float3
// template void apply_kernel<float3, float3>(
//     Array<float3>, LegDimensions,
//     void (*)(Array<float3>, LegDimensions, Array<float3>), Array<float3>);
//
// // Explicit instantiation for float3, bool
// template void apply_kernel<float3, bool>(Array<float3>, LegDimensions,
//                                          void (*)(const Array<float3>,
//                                                   LegDimensions, Array<bool>),
//                                          Array<bool>);

// template <typename T>
// Array<T> thustVectToArray(thrust::device_vector<T> thrust_vect) {
//     Array<T> out;
//     out.length = thrust_vect.size();
//     cudaMalloc(&out.elements, sizeof(T) * out.length);
//     cudaMemcpy(out.elements, thrust::raw_pointer_cast(thrust_vect.data()),
//                sizeof(T) * out.length, cudaMemcpyDeviceToDevice);
//     return out;
// }

template <typename T>
Array<T> thustVectToArray(thrust::host_vector<T> thrust_vect) {
    Array<T> out;
    out.length = thrust_vect.size();
    out.elements = new T[out.length];
    std::copy(thrust_vect.data(), thrust_vect.data() + thrust_vect.size(),
              out.elements);
    return out;
}
template Array<float3>
thustVectToArray<float3>(thrust::host_vector<float3> thrust_vect);
template Array<int> thustVectToArray<int>(thrust::host_vector<int> thrust_vect);

// template <typename T>
// thrust::device_vector<T> arrayToThrustVect(Array<T> array) {
//     thrust::device_vector<float3> vectOut(array.elements,
//                                           array.elements + array.length);
//     return vectOut;
// }
//
// template <typename T> thrust::host_vector<T> arrayToThrustVect(Array<T>
// array) {
//     thrust::host_vector<float3> vectOut(array.elements,
//                                         array.elements + array.length);
//     return vectOut;
// }

template <typename MyType>
__device__ MyType MinRowElement<MyType>::operator()(
    const thrust::tuple<MyType, MyType>& t) const {
    return thrust::min(thrust::get<0>(t), thrust::get<1>(t));
};
// template struct MinRowElement<float>;
template struct MinRowElement<int>;
template struct MinRowElement<unsigned char>;

__global__ void bring_together_kernel(unsigned char* outValidate,
                                 unsigned char* outEliminate, size_t N) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < N; i += stride) {
        bool res = (outValidate[i] == 1) && (outEliminate[i] != 1);
        outValidate[i] = res ? 1 : 0;
    }
}

// template <typename Tred, typename Tcheck, class F>
// __global__ void double_reduction_kernel(Tred* toReduce, size_t Nred,
//                                         Tcheck* toCheck, size_t Ncheck,
//                                         unsigned char* output,
//                                         F checkFunction) {
//     __shared__ bool result;
//     __shared__ float3 data;
//     auto data_index = blockIdx.x;
//     auto target_index = threadIdx.x;
//
//     if (target_index == 0) {
//         result = false;
//         data = toReduce[data_index];
//     }
//     __syncthreads();
//
//     if ((data_index < Nred) and (target_index < Ncheck)) {
//         const float3 target = toCheck[target_index];
//         if (checkFunction(data, target)) {
//             result = true;
//         }
//     }
//
//     __syncthreads();
//     if ((target_index == 0) && (result)) {
//         output[data_index] = 1;
//     }
// }
//
// template <typename Tred, typename Tcheck, class F>
// void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck* toCheck,
//                              const size_t Ncheck, unsigned char* output,
//                              // bool (*checkFunction)(Tred, Tcheck)) {
//                              F checkFunction) {
//     size_t processed_index = 0;
//     size_t max_block_size = 1024 / 1;
//     // size_t max_block_size = 1;
//     size_t numBlock = Nred;
//     while (processed_index < Ncheck) {
//         size_t targets_left_to_process = Ncheck - processed_index;
//         size_t blockSize = std::min(targets_left_to_process, max_block_size);
//         Tred* sub_target_ptr = toCheck + processed_index;
//         size_t sub_target_size = blockSize;
//
//         double_reduction_kernel<<<numBlock, blockSize>>>(
//             toReduce, Nred, sub_target_ptr, sub_target_size, output,
//             checkFunction);
//
//         processed_index += blockSize;
//     }
// }


