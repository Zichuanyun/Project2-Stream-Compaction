#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernEfficientGpuScan(int n, int *odata, const int *idata) {
        
        }

        __global__ void kernBuildTree(const int n, const int* d_data_in, int* d_data_out) {
          
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          dim3 blockSize(BLOCK_SIZE);
          dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

          // allocate memory
          int* d_data_in;
          int* d_data_out;
          cudaMalloc((void**)&d_data_in, n * sizeof(int));
          cudaMalloc((void**)&d_data_out, n * sizeof(int));
          cudaMemcpy(d_data_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
          timer().startGpuTimer();



          timer().endGpuTimer();
          // for readbility
          std::swap(d_data_in, d_data_out);
          odata[0] = 0;
          cudaMemcpy(odata + 1, d_data_out, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(d_data_out);
          cudaFree(d_data_in);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
