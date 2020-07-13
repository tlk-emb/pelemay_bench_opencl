#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// -----------
// use static data size
// #define DATA_SIZE (1024)
// source size of opencl kernel
#define MAX_SOURCE_SIZE 0x100000

// -----------

// int DATA_SIZE = 4096;
int DATA_SIZE = 65536;
int TEST_ITER = 5000;

void print_time(char *name, clock_t start, clock_t end)
{
	printf("%s took %f seconds\n", name, (double)(end - start) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv)
{
	int err;

	// read kernel code
	FILE *fp;
	char fileName[] = "./bench.cl";
	char *KernelSource;
	size_t source_size;
	fp = fopen(fileName, "r");
	if (!fp)
	{
		exit(1);
	}
	KernelSource = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(KernelSource, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// generate data for testing //

	// create random data with float
	int i = 0;
	unsigned int count = DATA_SIZE;
	float data[DATA_SIZE];
	for (i = 0; i < count; i++)
		// data[i] = rand() / (float)RAND_MAX;
		data[i] = (float)i;

	int data_int[DATA_SIZE];
	for (i = 0; i < count; i++)
	{
		// data_int[i] = (long int)(rand() / 1000);
		data_int[i] = (int)i;
	}

	// create random data with int
	long int data_long_int[DATA_SIZE];
	for (i = 0; i < count; i++)
	{
		// data_int[i] = (long int)(rand() / 1000);
		data_int[i] = (long int)i;
	}

	// Connect to a compute device
	int gpu = 1;
	cl_device_id device_id;
	err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// create command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// build program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	clock_t start_init, end_init;
	start_init = clock();

	// Create the compute kernel in the program we wish to run
	//
	clock_t kernel_start, kernel_end;
	kernel_start = clock();
	cl_kernel kernel_square = clCreateKernel(program, "square", &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}
	kernel_end = clock();
	print_time("kernel initialization", kernel_start, kernel_end);

	clock_t data_start, data_end;
	data_start = clock();

	// Create the input and output arrays in device memory for our calculation
	cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * count, NULL, NULL);
	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * count, NULL, NULL);

	if (!input || !output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(int) * count, data_int, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write to source array!\n");
		exit(1);
	}

	data_end = clock();
	print_time("data transfer", data_start, data_end);

	end_init = clock();
	print_time("opencl initialization", start_init, end_init);

	// ==================

	// Get the maximum work group size for executing the kernel on the device
	size_t local;
	err = clGetKernelWorkGroupInfo(kernel_square, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

	size_t global = count;

	clock_t default_start, default_end;
	default_start = clock();

	err = 0;
	// set arguments for square
	err = clSetKernelArg(kernel_square, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel_square, 1, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel_square, 2, sizeof(unsigned int), &count);

	err = clEnqueueNDRangeKernel(commands, kernel_square, 1, NULL, &global, &local, 0, NULL, NULL);

	// Read back the results from the device to verify the output
	//
	int results[DATA_SIZE]; // results returned from device
	err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(int) * count, results, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	default_end = clock();
	print_time("default", default_start, default_end);

	// ================================== //
	//	Check if the results are correct  //
	// ================================== //
	unsigned int correct = 0;
	for (i = 0; i < count; i++)
	{
		if (results[i] == data_int[i] * data_int[i])
			correct++;
	}

	printf("\n=== result summary ===\n");
	printf("Computed '%d/%d' correct values for square!\n", correct, count);

	// Shutdown and cleanup
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel_square);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}