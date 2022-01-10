
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;


#define THREADS_NO 512

cudaError_t laplaceFilterWithCuda(unsigned char* matrix, unsigned char* result_matrix, float* filter, int rows, int columns);

__global__ void laplaceFilterKernel(unsigned char* matrix, unsigned char* result, float* filter, int rows, int columns, int filter_rows, int filter_columns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * columns) {
        // get row and column index from liniarised matrix index
        int row = index / columns;
        int column = index % columns;

        // compute limits
        int upperLimit = row - filter_rows / 2;
        int lowerLimit = row + filter_rows / 2;
        int leftLimit = column - filter_columns / 2;
        int rightLimit = column + filter_columns / 2;

        int filterStartLine = 0;
        int filterEndLine = filter_rows;
        int filterStartColumn = 0;
        int filterEndColumn = filter_columns;

        // upper limit overflow
        if (upperLimit < 0) {
            filterStartLine = 0 - upperLimit;
            upperLimit = 0;
        }
        // lower limit overflow
        if (lowerLimit > rows - 1) {
            filterEndLine = filter_rows - (lowerLimit - rows + 1);
            lowerLimit = rows - 1;
        }
        // left limit overflow
        if (leftLimit < 0) {
            filterStartColumn = 0 - leftLimit;
            leftLimit = 0;
        }
        // right limit overflow
        if (rightLimit > columns - 1) {
            filterEndColumn = filter_columns - (rightLimit - columns + 1);
            rightLimit = columns - 1;
        }

        int filterLine = filterStartLine, filterColumn = filterStartColumn;
        float sum = 0;
        // overlap kernel over current element
        for (int i = upperLimit; i <= lowerLimit; i++) {
            for (int j = leftLimit; j <= rightLimit; j++) {
                sum += matrix[i * columns + j] * filter[filterLine * filter_columns + filterColumn];
                filterColumn++;
            }
            filterLine++;
            filterColumn = filterStartColumn;
        }
        // save computed result
        result[row * rows + column] = (char)sum / (filter_columns * filter_rows);
    }

}

// Laplacian filter - for determining edges
void filter_laplacian_init(float* filter) {
    filter[0] = 0;
    filter[1] = -1;
    filter[2] = 0;
    filter[3] = -1;
    filter[4] = 4;
    filter[5] = -1;
    filter[6] = 0;
    filter[7] = -1;
    filter[8] = 0;
}

void filter_blur_init(float* filter) {
    //float factor = 1;
    //filter[0] = factor * 0;
    //filter[1] = factor * -1;
    //filter[2] = factor * 0;
    //filter[3] = factor * -1;
    //filter[4] = factor * 5;
    //filter[5] = factor * -1;
    //filter[6] = factor * 0;
    //filter[7] = factor * -1;
    //filter[8] = factor * 0;
    float factor = 1;
    filter[0] = factor * 1;
    filter[1] = factor * 2;
    filter[2] = factor * 1;
    filter[3] = factor * 2;
    filter[4] = factor * 4;
    filter[5] = factor * 2;
    filter[6] = factor * 1;
    filter[7] = factor * 2;
    filter[8] = factor * 1;
}

int main()
{
    // declare input and output openCV matrixes
    cv::Mat inputImageRGBA;
    cv::Mat outputImageRGBA;

    // declare input and output values matrixes
    uchar4* inputImageRGBAMatrix;
    uchar4* outputImageRGBAMatrix;

    // input, output files
    string input_file{ "image.jpg" };
    string output_file{ "output.jpg" };
    string image_path = cv::samples::findFile(input_file);
    cv::cvtColor(cv::imread(image_path, 1), inputImageRGBA, 2);

    if (inputImageRGBA.empty()) {
        std::cerr << "Couldn't open file: " << input_file << std::endl;
        exit(1);
    }

    int numRows = inputImageRGBA.rows;
    int numCols = inputImageRGBA.cols;

    inputImageRGBAMatrix = new uchar4[numRows * numCols];
    outputImageRGBAMatrix = new uchar4[numRows * numCols];

    outputImageRGBA.create(numRows, numCols, CV_8UC4);

    memcpy(inputImageRGBAMatrix, (uchar4*)inputImageRGBA.ptr<unsigned char>(0), numRows * numCols * sizeof(uchar4));

    const size_t numPixels = numRows * numCols;

    unsigned char* grayscale = new unsigned char[numPixels];

    unsigned char* laplacian = new unsigned char[numPixels];

    float* filter = new float[3 * 3];

    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = inputImageRGBAMatrix[i];
        // convert to grayscale
        grayscale[i] = 0.2989 * rgba.x + 0.5870 * rgba.y + 0.1140 * rgba.z;
    }

    filter_laplacian_init(filter);
    cudaError_t cudaStatus = laplaceFilterWithCuda(grayscale, laplacian, filter, numRows, numCols);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "laplaceFilterWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = make_uchar4(laplacian[i], laplacian[i], laplacian[i], 255);
        outputImageRGBAMatrix[i] = rgba;
    }

    memcpy((uchar4*)outputImageRGBA.ptr<unsigned char>(0), outputImageRGBAMatrix, numRows * numCols * sizeof(uchar4));

    cv::Mat imageOutputRGB;
    cv::cvtColor(outputImageRGBA, imageOutputRGB, 3);
    cv::imwrite(output_file.c_str(), imageOutputRGB);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] grayscale;
    delete[] laplacian;

    return 0;
}

cudaError_t laplaceFilterWithCuda(unsigned char* matrix, unsigned char* result_matrix, float* filter, int rows, int columns)
{
    unsigned char* dev_matrix = 0;
    unsigned char* dev_result = 0;
    float* dev_filter = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_result, rows * columns * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_matrix, rows * columns * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_filter, 3 * 3 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_matrix, matrix, rows * columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_result, result_matrix, rows * columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_filter, filter, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int blocksNo = rows * columns / (THREADS_NO - 1);
    laplaceFilterKernel << <blocksNo, THREADS_NO >> > (dev_matrix, dev_result, dev_filter, rows, columns, 3, 3);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "laplaceFilterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching laplaceFilterKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(result_matrix, dev_result, rows * columns * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_matrix);
    cudaFree(dev_result);
    cudaFree(dev_filter);

    return cudaStatus;

}
