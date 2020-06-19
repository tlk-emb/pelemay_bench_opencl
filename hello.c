//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

#define MAX_SOURCE_SIZE 0x100000

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	int err; // error code returned from api calls

	float data[DATA_SIZE];				 // original data set given to device
	float results[DATA_SIZE];			 // results returned from device
	float results_vector_add[DATA_SIZE]; // results returned from device
	unsigned int correct;				 // number of correct results returned
	unsigned int correct_vector_add;	 // number of correct results returned

	size_t global; // global domain size for our calculation
	size_t local;  // local domain size for our calculation

	cl_device_id device_id;		 // compute device id
	cl_context context;			 // compute context
	cl_command_queue commands;	 // compute command queue
	cl_program program;			 // compute program
	cl_kernel kernel_square;	 // compute kernel
	cl_kernel kernel_vector_add; // compute kernel

	cl_mem input;			  // device memory used for the input array
	cl_mem output;			  // device memory used for the output array
	cl_mem output_vector_add; // device memory used for the output of vector_add

	// read kernel code
	FILE *fp;
	char fileName[] = "./test.cl";
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

	// Fill our data set with random float values
	//
	int i = 0;
	unsigned int count = DATA_SIZE;
	for (i = 0; i < count; i++)
		data[i] = rand() / (float)RAND_MAX;

	i = 0;
	for (i = 0; i < count; i++)
		data[i] = rand() / (int)RAND_MAX;

	// Connect to a compute device
	//
	int gpu = 1;
	err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	// Create a compute context
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	//
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	//
	program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	//
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

	// Create the compute kernel in the program we wish to run
	//
	kernel_square = clCreateKernel(program, "square", &err);
	kernel_vector_add = clCreateKernel(program, "vector_add", &err);
	if (!kernel_square || !kernel_vector_add || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the input and output arrays in device memory for our calculation
	//
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
	output_vector_add = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
	if (!input || !output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	// Write our data set into the input array in device memory
	//
	err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write to source array!\n");
		exit(1);
	}

	// Set the arguments to our compute kernel_square
	//
	err = 0;
	err = clSetKernelArg(kernel_square, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel_square, 1, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel_square, 2, sizeof(unsigned int), &count);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel_square arguments! %d\n", err);
		exit(1);
	}

	// Set the arguments to our compute kernel_vector_add
	err = 0;
	err = clSetKernelArg(kernel_vector_add, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel_vector_add, 1, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel_vector_add, 2, sizeof(cl_mem), &output_vector_add);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel_vector_add arguments! %d\n", err);
		exit(1);
	}

	// Get the maximum work group size for executing the kernel on the device
	//
	err = clGetKernelWorkGroupInfo(kernel_square, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	//
	global = count;
	err = clEnqueueNDRangeKernel(commands, kernel_square, 1, NULL, &global, &local, 0, NULL, NULL);
	err = clEnqueueNDRangeKernel(commands, kernel_vector_add, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}

	// Wait for the command commands to get serviced before reading back results
	//
	clFinish(commands);

	// Read back the results from the device to verify the output
	//
	err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
	err = clEnqueueReadBuffer(commands, output_vector_add, CL_TRUE, 0, sizeof(float) * count, results_vector_add, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	// Validate our results
	//
	correct = 0;
	for (i = 0; i < count; i++)
	{
		if (results[i] == data[i] * data[i])
			correct++;
	}

	correct_vector_add = 0;
	for (i = 0; i < count; i++)
	{
		if (results_vector_add[i] == data[i] + data[i])
		{
			correct_vector_add++;
		}
	}

	// Print a brief summary detailing the results
	//
	printf("Computed '%d/%d' correct values for square!\n", correct, count);
	printf("Computed '%d/%d' correct values for vector_add!\n", correct_vector_add, count);

	// Shutdown and cleanup
	//
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseMemObject(output_vector_add);
	clReleaseProgram(program);
	clReleaseKernel(kernel_square);
	clReleaseKernel(kernel_vector_add);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
