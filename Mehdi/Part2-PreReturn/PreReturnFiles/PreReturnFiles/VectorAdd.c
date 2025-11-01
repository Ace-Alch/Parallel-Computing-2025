/* COMP.CE.350 Vector Addition with OpenCL
   Copyright (c) 2024 Topi Leppanen  topi.leppanen@tuni.fi
*/

#include <stdio.h>  // printf
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120

#ifndef __APPLE__
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif

cl_context context;
cl_command_queue commandQueue;
cl_program program;
cl_kernel kernel;
cl_mem bufA;
cl_mem bufB;
cl_mem bufC;

const int PLATFORM_INDEX = 0;
const int DEVICE_INDEX = 0;

const char *openclErrors[] = {
    "Success!",
    "Device not found.",
    "Device not available",
    "Compiler not available",
    "Memory object allocation failure",
    "Out of resources",
    "Out of host memory",
    "Profiling information not available",
    "Memory copy overlap",
    "Image format mismatch",
    "Image format not supported",
    "Program build failure",
    "Map failure",
    "Invalid value",
    "Invalid device type",
    "Invalid platform",
    "Invalid device",
    "Invalid context",
    "Invalid queue properties",
    "Invalid command queue",
    "Invalid host pointer",
    "Invalid memory object",
    "Invalid image format descriptor",
    "Invalid image size",
    "Invalid sampler",
    "Invalid binary",
    "Invalid build options",
    "Invalid program",
    "Invalid program executable",
    "Invalid kernel name",
    "Invalid kernel definition",
    "Invalid kernel",
    "Invalid argument index",
    "Invalid argument value",
    "Invalid argument size",
    "Invalid kernel arguments",
    "Invalid work dimension",
    "Invalid work group size",
    "Invalid work item size",
    "Invalid global offset",
    "Invalid event wait list",
    "Invalid event",
    "Invalid operation",
    "Invalid OpenGL object",
    "Invalid buffer size",
    "Invalid mip-map level",
    "Unknown",
};


const char *clErrorString(cl_int e)
{
   switch (e) {
      case CL_SUCCESS:                            return openclErrors[ 0];
      case CL_DEVICE_NOT_FOUND:                   return openclErrors[ 1];
      case CL_DEVICE_NOT_AVAILABLE:               return openclErrors[ 2];
      case CL_COMPILER_NOT_AVAILABLE:             return openclErrors[ 3];
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return openclErrors[ 4];
      case CL_OUT_OF_RESOURCES:                   return openclErrors[ 5];
      case CL_OUT_OF_HOST_MEMORY:                 return openclErrors[ 6];
      case CL_PROFILING_INFO_NOT_AVAILABLE:       return openclErrors[ 7];
      case CL_MEM_COPY_OVERLAP:                   return openclErrors[ 8];
      case CL_IMAGE_FORMAT_MISMATCH:              return openclErrors[ 9];
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return openclErrors[10];
      case CL_BUILD_PROGRAM_FAILURE:              return openclErrors[11];
      case CL_MAP_FAILURE:                        return openclErrors[12];
      case CL_INVALID_VALUE:                      return openclErrors[13];
      case CL_INVALID_DEVICE_TYPE:                return openclErrors[14];
      case CL_INVALID_PLATFORM:                   return openclErrors[15];
      case CL_INVALID_DEVICE:                     return openclErrors[16];
      case CL_INVALID_CONTEXT:                    return openclErrors[17];
      case CL_INVALID_QUEUE_PROPERTIES:           return openclErrors[18];
      case CL_INVALID_COMMAND_QUEUE:              return openclErrors[19];
      case CL_INVALID_HOST_PTR:                   return openclErrors[20];
      case CL_INVALID_MEM_OBJECT:                 return openclErrors[21];
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return openclErrors[22];
      case CL_INVALID_IMAGE_SIZE:                 return openclErrors[23];
      case CL_INVALID_SAMPLER:                    return openclErrors[24];
      case CL_INVALID_BINARY:                     return openclErrors[25];
      case CL_INVALID_BUILD_OPTIONS:              return openclErrors[26];
      case CL_INVALID_PROGRAM:                    return openclErrors[27];
      case CL_INVALID_PROGRAM_EXECUTABLE:         return openclErrors[28];
      case CL_INVALID_KERNEL_NAME:                return openclErrors[29];
      case CL_INVALID_KERNEL_DEFINITION:          return openclErrors[30];
      case CL_INVALID_KERNEL:                     return openclErrors[31];
      case CL_INVALID_ARG_INDEX:                  return openclErrors[32];
      case CL_INVALID_ARG_VALUE:                  return openclErrors[33];
      case CL_INVALID_ARG_SIZE:                   return openclErrors[34];
      case CL_INVALID_KERNEL_ARGS:                return openclErrors[35];
      case CL_INVALID_WORK_DIMENSION:             return openclErrors[36];
      case CL_INVALID_WORK_GROUP_SIZE:            return openclErrors[37];
      case CL_INVALID_WORK_ITEM_SIZE:             return openclErrors[38];
      case CL_INVALID_GLOBAL_OFFSET:              return openclErrors[39];
      case CL_INVALID_EVENT_WAIT_LIST:            return openclErrors[40];
      case CL_INVALID_EVENT:                      return openclErrors[41];
      case CL_INVALID_OPERATION:                  return openclErrors[42];
      case CL_INVALID_GL_OBJECT:                  return openclErrors[43];
      case CL_INVALID_BUFFER_SIZE:                return openclErrors[44];
      case CL_INVALID_MIP_LEVEL:                  return openclErrors[45];
      default:                                    return openclErrors[46];
   }
}
// This function reads in a text file and stores it as a char pointer
char *
readSource(char *kernelPath) {
    cl_int status;
    FILE *fp;
    char *source;
    long int size;
    printf("Program file is: %s\n", kernelPath);
    fp = fopen(kernelPath, "rb");
    if (!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }
    rewind(fp);
    source = (char *)malloc(size + 1);
    if (source == NULL) {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }
    size_t readBytes = fread(source, 1, size, fp);
    if ((long int)readBytes != size) {
        printf("Error reading the kernel file\n");
        exit(-1);
    }
    source[size] = '\0';
    fclose(fp);
    return source;
}

// Informational printing
void
printPlatformInfo(cl_platform_id *platformId, size_t ret_num_platforms) {
    size_t infoLength = 0;
    char *infoStr = NULL;
    cl_int status;
    for (unsigned int r = 0; r < (unsigned int)ret_num_platforms; ++r) {
        printf("Platform %d information:\n", r);
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_PROFILE, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Platform profile length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_PROFILE, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Platform profile info error: %s\n", clErrorString(status));
        }
        printf("\tProfile: %s\n", infoStr);
        free(infoStr);
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VERSION, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Platform version length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VERSION, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Platform version info error: %s\n", clErrorString(status));
        }
        printf("\tVersion: %s\n", infoStr);
        free(infoStr);
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_NAME, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Platform name length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_NAME, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Platform name info error: %s\n", clErrorString(status));
        }
        printf("\tName: %s\n", infoStr);
        free(infoStr);
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VENDOR, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Platform vendor info length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_VENDOR, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Platform vendor info error: %s\n", clErrorString(status));
        }
        printf("\tVendor: %s\n", infoStr);
        free(infoStr);
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_EXTENSIONS, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Platform extensions info length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetPlatformInfo(platformId[r], CL_PLATFORM_EXTENSIONS, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Platform extensions info error: %s\n", clErrorString(status));
        }
        printf("\tExtensions: %s\n", infoStr);
        free(infoStr);
    }
    printf("\nUsing Platform %d.\n", PLATFORM_INDEX);
}

// Informational printing
void
printDeviceInfo(cl_device_id *deviceIds, size_t ret_num_devices) {
    // Print info about the devices
    size_t infoLength = 0;
    char *infoStr = NULL;
    cl_int status;

    for (unsigned int r = 0; r < ret_num_devices; ++r) {
        printf("Device %d indormation:\n", r);
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VENDOR, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Device Vendor info length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VENDOR, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Device Vendor info error: %s\n", clErrorString(status));
        }
        printf("\tVendor: %s\n", infoStr);
        free(infoStr);
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_NAME, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Device name info length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_NAME, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Device name info error: %s\n", clErrorString(status));
        }
        printf("\tName: %s\n", infoStr);
        free(infoStr);
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VERSION, 0, NULL, &infoLength);
        if (status != CL_SUCCESS) {
            printf("Device version info length error: %s\n", clErrorString(status));
        }
        infoStr = malloc((infoLength) * sizeof(char));
        status = clGetDeviceInfo(deviceIds[r], CL_DEVICE_VERSION, infoLength, infoStr, NULL);
        if (status != CL_SUCCESS) {
            printf("Device version info error: %s\n", clErrorString(status));
        }
        printf("\tVersion: %s\n", infoStr);
        free(infoStr);
    }
    printf("\nUsing Device %d.\n", DEVICE_INDEX);
}

int
main() {
    // Initialize RNG
    srand(42);

    const int elements = 16;
    size_t datasize = sizeof(int) * elements;

    // Host data (allocated on the host memory/CPU)
    int *A = (int *)malloc(datasize);
    int *B = (int *)malloc(datasize);
    for (int i = 0; i < elements; i++) {
        A[i] = rand() % 64;
        B[i] = rand() % 64;
    }
    int *C = (int *)malloc(datasize);  // Output array

    // Start the OpenCL initialization
    cl_int status;  // Use this to check the output of each API call

    // Get available OpenCL platforms
    cl_uint ret_num_platforms;
    status = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    if (status != CL_SUCCESS) {
        printf("Error getting the number of platforms: %s", clErrorString(status));
    }
    cl_platform_id *platformId = malloc(sizeof(cl_platform_id) * ret_num_platforms);
    status = clGetPlatformIDs(ret_num_platforms, platformId, NULL);
    if (status != CL_SUCCESS) {
        printf("Error getting the platforms: %s", clErrorString(status));
    }

    // Print info about the platform. Not needed for functionality,
    // but nice to see in order to confirm your OpenCL installation
    printPlatformInfo(platformId, ret_num_platforms);

    // Get available devices
    cl_uint ret_num_devices = 0;
    status = clGetDeviceIDs(
        platformId[PLATFORM_INDEX], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
    if (status != CL_SUCCESS) {
        printf("Error getting the number of devices: %s", clErrorString(status));
    }
    cl_device_id *deviceIds = malloc((ret_num_devices) * sizeof(cl_device_id));
    status = clGetDeviceIDs(
        platformId[PLATFORM_INDEX], CL_DEVICE_TYPE_ALL, ret_num_devices, deviceIds, &ret_num_devices);
    if (status != CL_SUCCESS) {
        printf("Error getting device ids: %s", clErrorString(status));
    }

    // Again, this only prints nice-to-know information
    printDeviceInfo(deviceIds, ret_num_devices);

    context = clCreateContext(NULL, 1, &(deviceIds[DEVICE_INDEX]), NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Context creation error: %s\n", clErrorString(status));
    }

    // In order command queue
    // Using the 1.2 clCreateCommandQueue API since it's bit simpler,
    // this was later deprecated in OpenCL 2.0
    commandQueue = clCreateCommandQueue(context, deviceIds[DEVICE_INDEX], 0, &status);
    if (status != CL_SUCCESS) {
        printf("Command queue creation error: %s", clErrorString(status));
    }

    // Make kernel string into a program
    const char *programSource = readSource("VectorAdd.cl");
    program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Program creation error: %s", clErrorString(status));
    }

    // Program compiling
    status = clBuildProgram(program, 1, &deviceIds[DEVICE_INDEX], NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("OpenCL build error: %s\n", clErrorString(status));
        // Fetch build errors if there were some.
        if (status == CL_BUILD_PROGRAM_FAILURE) {
            size_t infoLength = 0;
            cl_int cl_build_status = clGetProgramBuildInfo(
                program, deviceIds[DEVICE_INDEX], CL_PROGRAM_BUILD_LOG, 0, 0, &infoLength);
            if (cl_build_status != CL_SUCCESS) {
                printf("Build log length fetch error: %s\n", clErrorString(cl_build_status));
            }
            char *infoStr = malloc(infoLength * sizeof(char));
            cl_build_status = clGetProgramBuildInfo(
                program, deviceIds[DEVICE_INDEX], CL_PROGRAM_BUILD_LOG, infoLength, infoStr, 0);
            if (cl_build_status != CL_SUCCESS) {
                printf("Build log fetch error: %s\n", clErrorString(cl_build_status));
            }

            printf("OpenCL build log:\n %s", infoStr);
            free(infoStr);
        }
        abort();
    }

    // Create the vector addition kernel
    kernel = clCreateKernel(program, "vecadd", &status);
    if (status != CL_SUCCESS) {
        printf("Kernel creation error: %s\n", clErrorString(status));
    }

    // Create a buffer object that will contain the data
    // from the host array A
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Buffer A creation error: %s", clErrorString(status));
    }

    // Create a buffer object that will contain the data
    // from the host array B
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Buffer B creation error: %s", clErrorString(status));
    }

    // Create a buffer object that will hold the output data
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);
    if (status != CL_SUCCESS) {
        printf("Buffer C creation error: %s", clErrorString(status));
    }

    // Write input array A to the device buffer bufferA
    status = clEnqueueWriteBuffer(commandQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Buffer A write error: %s", clErrorString(status));
    }

    // Write input array B to the device buffer bufferB
    status = clEnqueueWriteBuffer(commandQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Buffer B write error: %s", clErrorString(status));
    }

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    if (status != CL_SUCCESS) {
        printf("Error setting arg 0: %s", clErrorString(status));
    }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    if (status != CL_SUCCESS) {
        printf("Error setting arg 1: %s", clErrorString(status));
    }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    if (status != CL_SUCCESS) {
        printf("Error setting arg 2: %s", clErrorString(status));
    }

    // Define an index space (global work size) of work
    // items for execution. A workgroup size (local work size)
    // is not required, but can be used.
    size_t globalWorkSize[1];
    // There are 'elements' work-items
    globalWorkSize[0] = elements;

    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Kernel enqueuing error%s", clErrorString(status));
    }

    // Read the device output buffer to the host output array.
    // Blocking read.
    status = clEnqueueReadBuffer(commandQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Buffer C read error%s", clErrorString(status));
    }

    // Print the input and output arrays, and validate the output
    for (int i = 0; i < elements; i++) {
        printf("%-2d + %-2d = %-3d", A[i], B[i], C[i]);
        if ((A[i] + B[i]) == C[i]) {
            printf(" OK\n");
        } else {
            printf(" WRONG\n");
        }
    }

    // Release all OpenCL objects that we created ourselves
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
}
