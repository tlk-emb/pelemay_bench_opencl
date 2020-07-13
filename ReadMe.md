# pelemay_bench_opencl

OpenCL implementation of [`pelemay_sample`](https://github.com/zeam-vm/pelemay_sample). 

Some code derived from [Apple OpenCL HelloWorld Example](https://developer.apple.com/library/archive/samplecode/OpenCL_Hello_World_Example/Introduction/Intro.html).

## `opencl_bench`

+ Length of testing array can be changed by changing `DATA_SIZE = ;` in each file.

`bench.cl`
: Contains kernels for benchmarks.

`square_bench.c`
: Executes `Square` benchmark.

`logistic_map_10_bench.c`
: Executes `LogisticMap10` benchmark.

`logistic_map_20_bench.c`
: Executes `LogisticMap20` benchmark.

### Execute

Run `make all` in `opencl_bench/`.

## pelemay bench

Code derived from `pelemay_sample`.

### Execute

Run `mix bench` in `pelemay_bench/`.
