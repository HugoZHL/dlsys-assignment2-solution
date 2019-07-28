#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */
__global__ void array_set_kernel(int n, float *arr, float value) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n) {
    return;
  }
  arr[y] = value;
}

__global__ void broadcast_to_kernel(int in_arr, int out_arr, const float *input, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= in_arr) {
    return;
  }
  for (int i = y; i < out_arr; i += in_arr) {
    output[i] = input[y];
  }
}

__global__ void reduce_sum_axis_zero_kernel(int in_arr, int out_arr, const float *input, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= out_arr) {
    return;
  }
  output[y] = 0;
  for (int i = y; i < in_arr; i += out_arr) {
    output[y] += input[i];
  }
}

__global__ void matrix_add_kernel(int longer, int shorter, 
                                  const float *matA, 
                                  const float *matB, 
                                  float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= shorter) {
    return;
  }
  for (int i = y; i < longer; i += shorter) {
    output[i] = matA[i] + matB[y];
  }
}

__global__ void matrix_add_by_const_kernel(int n, const float *mat, float val, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n) {
    return;
  }
  output[y] = mat[y] + val;
}

__global__ void matrix_mul_kernel(int longer, int shorter, 
                                  const float *matA, 
                                  const float *matB, 
                                  float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= shorter) {
    return;
  }
  for (int i = y; i < longer; i += shorter) {
    output[i] = matA[i] * matB[y];
  }
}

__global__ void matrix_mul_by_const_kernel(int n, const float *mat, float val, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n) {
    return;
  }
  output[y] = mat[y] * val;
}

__global__ void relu_kernel(int n, const float *input, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n) {
    return;
  }
  output[y] = (input[y] > 0.0 ? input[y] : 0.0);
}

__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  output += y * ncol;
  float maxval = *input;
  for (int i = 1; i < ncol; ++i) {
    maxval = max(maxval, input[i]);
  }
  float sum = 0;
  for (int i = 0; i < ncol; ++i) {
    sum += exp(input[i] - maxval);
  }
  for (int i = 0; i < ncol; ++i) {
    output[i] = exp(input[i] - maxval) / sum;
  }
}

__global__ void relu_gradient_kernel(int n, const float *input, const float *grad, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= n) {
    return;
  }
  output[y] = input[y] > 0.0 ? grad[y] : 0.0;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int n = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    n *= arr->shape[i];
  }
  float *array_data = (float *) arr->data;
  dim3 threads, blocks;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  array_set_kernel<<<blocks, threads>>>(n, array_data, value);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(output->ndim >= input->ndim);
  int in_arr = 1, out_arr = 1;
  int diff = output->ndim - input->ndim;
  for (int i = 0; i < diff; ++i) {
    out_arr *= output->shape[i];
  }
  for (int i = 0; i < input->ndim; ++i) {
    assert(input->shape[i] == output->shape[i+diff]);
    in_arr *= input->shape[i];
    out_arr *= input->shape[i];
  }
  dim3 threads, blocks;
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  if (in_arr <= 1024) {
    threads.x = in_arr;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (in_arr + 1023) / 1024;
  }
  broadcast_to_kernel<<<blocks, threads>>>(in_arr, out_arr, input_data, output_data);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim >= output->ndim);
  int in_arr = 1, out_arr = 1;
  int diff = input->ndim - output->ndim;
  for (int i = 0; i < diff; ++i) {
    in_arr *= input->shape[i];
  }
  for (int i = 0; i < output->ndim; ++i) {
    assert(input->shape[i+diff] == output->shape[i]);
    in_arr *= output->shape[i];
    out_arr *= output->shape[i];
  }
  dim3 threads, blocks;
  if (out_arr <= 1024) {
    threads.x = out_arr;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (out_arr + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(in_arr, out_arr, input_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  if (matA->ndim < matB->ndim) {
    return DLGpuMatrixElementwiseAdd(matB, matA, output);
  }
  int cntA = 1, cntB = 1;
  int diff = matA->ndim - matB->ndim;
  for (int i = 0; i < diff; ++i) {
    cntA *= matA->shape[i];
  }
  for (int i = 0; i < matB->ndim; ++i) {
    assert(matA->shape[i+diff] == matB->shape[i]);
    cntA *= matB->shape[i];
    cntB *= matB->shape[i];
  }
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *) output->data;
  dim3 blocks, threads;
  if (cntA <= 1024) {
    threads.x = cntA;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (cntA + 1023) / 1024;
  }
  matrix_add_kernel<<<blocks, threads>>>(cntA, cntB, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  int matCnt = 1;
  for (int i = 0; i < input->ndim; ++i) {
    matCnt *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (matCnt <= 1024) {
    threads.x = matCnt;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (matCnt + 1023) / 1024;
  }
  matrix_add_by_const_kernel<<<blocks, threads>>>(matCnt, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  if (matA->ndim < matB->ndim) {
    return DLGpuMatrixElementwiseMultiply(matB, matA, output);
  }
  int cntA = 1, cntB = 1;
  int diff = matA->ndim - matB->ndim;
  for (int i = 0; i < diff; ++i) {
    cntA *= matA->shape[i];
  }
  for (int i = 0; i < matB->ndim; ++i) {
    assert(matA->shape[i+diff] == matB->shape[i]);
    cntA *= matB->shape[i];
    cntB *= matB->shape[i];
  }
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *output_data = (float *) output->data;
  dim3 blocks, threads;
  if (cntA <= 1024) {
    threads.x = cntA;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (cntA + 1023) / 1024;
  }
  matrix_mul_kernel<<<blocks, threads>>>(cntA, cntB, matA_data, matB_data, output_data);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int matCnt = 1;
  for (int i = 0; i < input->ndim; ++i) {
    matCnt *= input->shape[i];
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (matCnt <= 1024) {
    threads.x = matCnt;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (matCnt + 1023) / 1024;
  }
  matrix_mul_by_const_kernel<<<blocks, threads>>>(matCnt, input_data, val, output_data);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2);
  assert(matB->ndim == 2);
  assert(matC->ndim == 2);
  cublasHandle_t handle;
  assert(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
  cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transposeA ? matA->shape[1] : matA->shape[0];
  int k1 = transposeA ? matA->shape[0] : matA->shape[1];
  int k2 = transposeB ? matB->shape[1] : matB->shape[0];
  int n = transposeB ? matB->shape[0] : matB->shape[1];
  assert(k1 == k2);
  assert(m == matC->shape[0]);
  assert(n == matC->shape[1]);
  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;
  float alpha = 1.0f, beta = 0.0f;
  DLGpuArraySet(matC, 0.0f);
  cublasSgemm(handle, transb, transa, 
              n, m, k1, &alpha, 
              matB_data, matB->shape[1], 
              matA_data, matA->shape[1], 
              &beta, matC_data, n);
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int n = input->shape[0] * input->shape[1];
  dim3 blocks, threads;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  relu_kernel<<<blocks, threads>>>(n, input_data, output_data);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(in_grad->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == in_grad->shape[0] &&
         input->shape[1] == in_grad->shape[1]);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int n = input->shape[0] * input->shape[1];
  dim3 blocks, threads;
  if (n <= 1024) {
    threads.x = n;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (n + 1023) / 1024;
  }
  const float *input_data = (const float *)input->data;
  const float *grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;
  relu_gradient_kernel<<<blocks, threads>>>(n, input_data, grad_data, output_data);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 blocks, threads;
  if (nrow <= 1024) {
    threads.x = nrow;
    blocks.x = 1;
  } else {
    threads.x = 1024;
    blocks.x = (nrow + 1023) / 1024;
  }
  softmax_kernel<<<blocks, threads>>>(nrow, ncol, input_data, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
