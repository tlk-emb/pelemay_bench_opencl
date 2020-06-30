__kernel void square(__global float *input, __global float *output,
                     const unsigned int count) {
  int i = get_global_id(0);
  if (i < count)
    output[i] = input[i] * input[i];
};

__kernel void vector_add(__global const float *a, __global const float *b,
                         __global float *c) {
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}

__kernel void logistic_map_long_int(__global const long int *a,
                                    __global long int *b) {
  int i = get_global_id(0);
  b[i] = (22 * a[i] * (a[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
}

__kernel void logistic_map_int(__global const int *a, __global int *b) {
  int i = get_global_id(0);
  b[i] = (22 * a[i] * (a[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
}