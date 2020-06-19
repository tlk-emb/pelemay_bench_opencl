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

__kernel void logistic_map(__global const int *a, __global int *b) {
  int i = get_global_id(0);
  b[i] = (22 * a[i] * (a[i] + 1)) % 6700417;
}
