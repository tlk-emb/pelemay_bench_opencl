__kernel void square(__global const int *input, __global int *output,
                     unsigned int count) {
  int i = get_global_id(0);
  if (count > i)
    output[i] = input[i] * input[i];
};

__kernel void logistic_map_10(__global const int *a, __global int *b) {
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

__kernel void logistic_map_20(__global const int *a, __global int *b) {
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
  b[i] = (22 * b[i] * (b[i] + 1)) % 6700417;
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