#include "math_util.h"
#include "HeaderCUDA.h"
#include "vector_types.h"
#include <fstream>
#include <iostream>

int calculateMedian(const Eigen::VectorXi& data) {
    // Convert the Eigen vector to a std::vector for sorting
    std::vector<int> stdVector(data.data(), data.data() + data.size());

    // Sort the std::vector
    std::sort(stdVector.begin(), stdVector.end());

    // Calculate the median
    int median;
    size_t size = stdVector.size();
    median = stdVector[size / 2]; // not very true but I don't care

    return median;
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

bool close(float a, float b, float interval) {
    return (a - interval < b) && (b < a + interval);
}

template <typename T>
void saveArrayToFile(T* array, long length, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(array), length * sizeof(T));
        file.close();
    } else {
        std::cout << "error saving file";
    }
}

template void saveArrayToFile<int>(int* array, long length,
                                   const char* filename);
template void saveArrayToFile<float>(float* array, long length,
                                     const char* filename);
template void saveArrayToFile<bool>(bool* array, long length,
                                     const char* filename);

template <typename T> Array<T> readArrayFromFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Determine the length of the file
        file.seekg(0, std::ios::end);
        long length = file.tellg() / sizeof(T);
        file.seekg(0, std::ios::beg);

        // Allocate memory for the array
        Array<T> arr;
        arr.length = length;
        arr.elements = new T[arr.length];

        // Read data from the file into the array
        file.read(reinterpret_cast<char*>(arr.elements), length * sizeof(T));

        file.close();

        return arr;
    } else {
        Array<T> arr;
        std::cerr << "Error opening file: " << filename << std::endl;
        return arr;
    }
}

template Array<float> readArrayFromFile<float>(const char* filename);

template <typename T>
Array<float3> threeArrays2float3Arr(Array<T> x, Array<T> y, Array<T> z) {
    int N = x.length;
    Array<float3> out;
    out.length = N;
    out.elements = new float3[out.length]();

    for (int i = 0; i < N; i++) {
        out.elements[i].x = x.elements[i];
        out.elements[i].y = y.elements[i];
        out.elements[i].z = z.elements[i];
    }
    return out;
}

template Array<float3>
threeArrays2float3Arr<float>(Array<float> x, Array<float> y, Array<float> z);

void savef3Arrayto3files(Array<float3> array_to_save, const char* filename) {
    {
        const char *str2 = "x";
        const char *str3 = ".bin";
        float* z_arr = new float[array_to_save.length];
        for (int i = 0; i < array_to_save.length; i++) {
            z_arr[i] = array_to_save.elements[i].x;
        }

        char* result =
            (char*)malloc(strlen(filename) + strlen(str2) + strlen(str3) +
                          1); // Allocate memory for concatenated string
        strcpy(result, filename); // Copy str1 to result
        strcat(result, str2); // Concatenate str2 to result
        strcat(result, str3); // Concatenate str2 to result

        // filename = "cpp_array_xz.bin";
        saveArrayToFile(z_arr, array_to_save.length, filename);
        delete[] z_arr;
        delete[] result;
    }
    {
        const char *str2 = "y";
        const char *str3 = ".bin";
        float* z_arr = new float[array_to_save.length];
        for (int i = 0; i < array_to_save.length; i++) {
            z_arr[i] = array_to_save.elements[i].y;
        }

        char* result =
            (char*)malloc(strlen(filename) + strlen(str2) + strlen(str3) +
                          1); // Allocate memory for concatenated string
        strcpy(result, filename); // Copy str1 to result
        strcat(result, str2); // Concatenate str2 to result
        strcat(result, str3); // Concatenate str2 to result

        // filename = "cpp_array_xz.bin";
        saveArrayToFile(z_arr, array_to_save.length, filename);
        delete[] z_arr;
        delete[] result;
    }
    {
        const char *str2 = "z";
        const char *str3 = ".bin";
        float* z_arr = new float[array_to_save.length];
        for (int i = 0; i < array_to_save.length; i++) {
            z_arr[i] = array_to_save.elements[i].z;
        }

        char* result =
            (char*)malloc(strlen(filename) + strlen(str2) + strlen(str3) +
                          1); // Allocate memory for concatenated string
        strcpy(result, filename); // Copy str1 to result
        strcat(result, str2); // Concatenate str2 to result
        strcat(result, str3); // Concatenate str2 to result

        // filename = "cpp_array_xz.bin";
        saveArrayToFile(z_arr, array_to_save.length, filename);
        delete[] z_arr;
        delete[] result;
    }
}
