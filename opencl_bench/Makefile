CC = gcc

.DEFAULT_TARGET = bench

bench: bench.c
	$(CC) $^ -framework opencl -o $@

all:
	$(CC) bench.c -framework opencl -o bench
	$(CC) square_bench.c -framework opencl -o square_bench
	$(CC) logistic_map_10_bench.c -framework opencl -o logistic_map_10_bench
	$(CC) logistic_map_20_bench.c -framework opencl -o logistic_map_20_bench
