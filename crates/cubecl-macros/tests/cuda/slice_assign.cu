typedef unsigned int uint;

extern "C" __global__ void kernel(float input_0[], float output_0[],
                                  uint info[]) {

  int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                        threadIdx.z * (blockDim.x * blockDim.y);
  uint rank = info[0];
  uint rank_2 = rank * 2;
  bool l_0_0;
  float l_0_1;
  l_0_0 = threadIdxGlobal == uint(0);
  if (l_0_0) {
    uint slice_1_0_length = uint(3) - uint(2);
    float *slice_1_0 = output_0 + uint(2);
    uint l_1_0;
    bool l_1_1;
    l_1_0 = info[(2 * 2 * info[0]) + 1];
    l_1_1 = uint(0) < l_1_0;
    if (l_1_1) {
      l_0_1 = input_0[uint(0)];
    } else {
      l_0_1 = float(0.0);
    }
    uint l_1_2;
    bool l_1_3;
    l_1_2 = slice_1_0_length;
    l_1_3 = uint(0) < l_1_2;
    if (l_1_3) {
      slice_1_0[uint(0)] = l_0_1;
    }
  }
}