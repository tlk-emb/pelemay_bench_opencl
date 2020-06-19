CC = gcc
CFLAGS = 

.DEFAULT_TARGET = test

test: test.c
	$(CC) $^ -framework opencl -o $@
