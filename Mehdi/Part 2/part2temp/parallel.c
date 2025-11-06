/* COMP.CE.350 Parallelization Excercise 2024
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
                      Topi Leppanen  topi.leppanen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
VERSION 24.0 - port to SDL2
VERSION 25.0 - add macOS support
*/



#ifdef _WIN32
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;    // NVIDIA
#endif



#ifdef _WIN32
#include "SDL.h"
#elif defined(__APPLE__)
#include "SDL.h"
#else
#include "SDL2/SDL.h"
#endif

#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

int mousePosX;
int mousePosY;

// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1920
#define SIZE WINDOW_WIDTH*WINDOW_HEIGHT

// The number of satellites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELLITE_COUNT 64

// These are used to control the satellite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000
#define BLACK_HOLE_RADIUS 4.5f


////////////////////////////////////////////////
//         ¤¤ ADDED OPENCL HANDLES ¤¤         //
////////////////////////////////////////////////
// --- OpenCL handles ---
static cl_platform_id      clPlat   =   NULL;
static cl_device_id        clDev    =   NULL;
static cl_context          clCtx    =   NULL;
static cl_command_queue    clQ      =   NULL;
static cl_program          clProg   =   NULL;
static cl_kernel           clKer    =   NULL;

static cl_mem              d_pixels =   NULL;
static cl_mem              d_pos_x  =   NULL;
static cl_mem              d_pos_y  =   NULL;
static cl_mem              d_id_r   =   NULL;
static cl_mem              d_id_g   =   NULL;
static cl_mem              d_id_b   =   NULL;

// Work-group size (edit these to test 1x1, 4x4, 8x4, 8x8, 16x16)
static size_t              WGX      =   16;
static size_t              WGY      =   16;


////////////////////////////////////////////////
//   ¤¤ LOAD KERNEL FILE USING MALLOC ¤¤      //
////////////////////////////////////////////////

#define CL_CHECK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", _e, __FILE__, __LINE__); exit(1);} } while(0)

static char* loadTextFile(const char* path, size_t* outSize) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f); fclose(f);
    buf[len] = '\0';
    if (outSize) *outSize = (size_t)len;
    return buf;
}





// Stores 2D data like the coordinates
typedef struct{
   float x;
   float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct{
   double x;
   double y;
} doublevector;

// Each float may vary from 0.0f ... 1.0f
typedef struct{
   float blue;
   float green;
   float red;
} color_f32;

// Stores rendered colors. Each value may vary from 0 ... 255
typedef struct{
   uint8_t blue;
   uint8_t green;
   uint8_t red;
   uint8_t reserved;
} color_u8;

// Stores the satellite data, which fly around black hole in the space
typedef struct{
   color_f32 identifier;
   floatvector position;
   floatvector velocity;
} satellite;

// Pixel buffer which is rendered to the screen
color_u8* pixels;

// Pixel buffer which is used for error checking
color_u8* correctPixels;

// Buffer for all satellites in the space
satellite* satellites;
satellite* backupSatelites;

////////////////////////////////////////////////
//   ¤¤     NVIDIA PICKER AS GPU     ¤¤       //
////////////////////////////////////////////////
// --- OpenCL device picker (prefer NVIDIA GPU) ---
static void pickOpenCLDevice(void) {
    cl_uint nplat = 0; CL_CHECK(clGetPlatformIDs(0, NULL, &nplat));
    if (!nplat) { fprintf(stderr, "No OpenCL platforms.\n"); exit(1); }
    cl_platform_id plats[16]; if (nplat > 16) nplat = 16;
    CL_CHECK(clGetPlatformIDs(nplat, plats, NULL));

    cl_platform_id chosenPlat = NULL;
    cl_device_id   chosenDev = NULL;

    // 1) NVIDIA GPU
    for (cl_uint p = 0; p < nplat && !chosenDev; ++p) {
        char vendor[256] = { 0 };
        clGetPlatformInfo(plats[p], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        if (strstr(vendor, "NVIDIA")) {
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &chosenDev, NULL) == CL_SUCCESS) {
                chosenPlat = plats[p]; break;
            }
        }
    }
    // 2) any GPU
    for (cl_uint p = 0; p < nplat && !chosenDev; ++p) {
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &chosenDev, NULL) == CL_SUCCESS) {
            chosenPlat = plats[p]; break;
        }
    }
    // 3) CPU fallback
    if (!chosenDev) {
        for (cl_uint p = 0; p < nplat && !chosenDev; ++p) {
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, 1, &chosenDev, NULL) == CL_SUCCESS) {
                chosenPlat = plats[p]; break;
            }
        }
    }
    if (!chosenDev) { fprintf(stderr, "No suitable OpenCL device.\n"); exit(1); }

    clPlat = chosenPlat; clDev = chosenDev;

    char platName[256] = { 0 }, platVendor[256] = { 0 }, devName[256] = { 0 };
    cl_device_type dtype = 0;
    clGetPlatformInfo(clPlat, CL_PLATFORM_NAME, sizeof(platName), platName, NULL);
    clGetPlatformInfo(clPlat, CL_PLATFORM_VENDOR, sizeof(platVendor), platVendor, NULL);
    clGetDeviceInfo(clDev, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
    clGetDeviceInfo(clDev, CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);

    printf("OpenCL platform: %s | vendor: %s\n", platName, platVendor);
    printf("OpenCL device  : %s | type: %s\n",
        devName,
        (dtype == CL_DEVICE_TYPE_GPU ? "GPU" :
            dtype == CL_DEVICE_TYPE_CPU ? "CPU" :
            dtype == CL_DEVICE_TYPE_ACCELERATOR ? "ACCEL" : "OTHER"));
}



// ## You may add your own variables here ##

// ## You may add your own initialization routines here ##
void init(){
    // Pick device first
    pickOpenCLDevice();

    cl_int err;
    // Context + queue
#if defined(CL_VERSION_2_0)
    const cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    clCtx = clCreateContext(NULL, 1, &clDev, NULL, NULL, &err); CL_CHECK(err);
    clQ = clCreateCommandQueueWithProperties(clCtx, clDev, props, &err); CL_CHECK(err);
#else
    clCtx = clCreateContext(NULL, 1, &clDev, NULL, NULL, &err); CL_CHECK(err);
    clQ = clCreateCommandQueue(clCtx, clDev, 0, &err); CL_CHECK(err);
#endif

    // Program + kernel
    size_t srcLen = 0;
    char* src = loadTextFile("parallel.cl", &srcLen);
    if (!src) { fprintf(stderr, "Could not load parallel.cl\n"); exit(1); }
    const char* srcs[] = { src }; const size_t lens[] = { srcLen };
    clProg = clCreateProgramWithSource(clCtx, 1, srcs, lens, &err); CL_CHECK(err);
    free(src);

    const char* buildOpts = "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations";
    err = clBuildProgram(clProg, 1, &clDev, buildOpts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize = 0; clGetProgramBuildInfo(clProg, clDev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = (char*)malloc(logSize + 1);
        clGetProgramBuildInfo(clProg, clDev, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        log[logSize] = '\0'; fprintf(stderr, "Build failed:\n%s\n", log); free(log);
        CL_CHECK(err);
    }
    clKer = clCreateKernel(clProg, "shade", &err); CL_CHECK(err);

    // Buffers
    // pixels: write directly into host memory
    d_pixels = clCreateBuffer(
        clCtx, CL_MEM_WRITE_ONLY,
        sizeof(unsigned char) * 4 * SIZE,
        NULL, &err);
    CL_CHECK(err);

    d_pos_x = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, SATELLITE_COUNT * sizeof(float), NULL, &err); CL_CHECK(err);
    d_pos_y = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, SATELLITE_COUNT * sizeof(float), NULL, &err); CL_CHECK(err);
    d_id_r = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, SATELLITE_COUNT * sizeof(float), NULL, &err); CL_CHECK(err);
    d_id_g = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, SATELLITE_COUNT * sizeof(float), NULL, &err); CL_CHECK(err);
    d_id_b = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, SATELLITE_COUNT * sizeof(float), NULL, &err); CL_CHECK(err);

    // Upload constant identifier colors once
    float h_id_r[SATELLITE_COUNT], h_id_g[SATELLITE_COUNT], h_id_b[SATELLITE_COUNT];
    for (int j = 0; j < SATELLITE_COUNT; ++j) {
        h_id_r[j] = satellites[j].identifier.red;
        h_id_g[j] = satellites[j].identifier.green;
        h_id_b[j] = satellites[j].identifier.blue;
    }
    CL_CHECK(clEnqueueWriteBuffer(clQ, d_id_r, CL_TRUE, 0, sizeof(h_id_r), h_id_r, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(clQ, d_id_g, CL_TRUE, 0, sizeof(h_id_g), h_id_g, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(clQ, d_id_b, CL_TRUE, 0, sizeof(h_id_b), h_id_b, 0, NULL, NULL));

    // Optional: print WG preference
    size_t pref = 0, maxWG = 0;
    clGetKernelWorkGroupInfo(clKer, clDev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(pref), &pref, NULL);
    clGetKernelWorkGroupInfo(clKer, clDev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxWG), &maxWG, NULL);
    printf("Preferred WG multiple: %zu | Max WG size: %zu\n", pref, maxWG);
}



// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine)
// Moves the satellites based on gravity
// This is done multiple times in a frame because the Euler integration
// is not accurate enough to be done only once
void parallelPhysicsEngine(void) {

    int tmpMousePosX = mousePosX;
    int tmpMousePosY = mousePosY;

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    doublevector tmpPosition[SATELLITE_COUNT];
    doublevector tmpVelocity[SATELLITE_COUNT];

    // Copy in (float -> double) once
    for (int idx = 0; idx < SATELLITE_COUNT; ++idx) {
        tmpPosition[idx].x = satellites[idx].position.x;
        tmpPosition[idx].y = satellites[idx].position.y;
        tmpVelocity[idx].x = satellites[idx].velocity.x;
        tmpVelocity[idx].y = satellites[idx].velocity.y;
    }

    const double dt = (double)DELTATIME / (double)PHYSICSUPDATESPERFRAME;

    int i;
#pragma omp parallel for schedule(static) // or: schedule(static, 8)
    for (i = 0; i < SATELLITE_COUNT; ++i) {

        // Work in registers to avoid false sharing
        double x = tmpPosition[i].x;
        double y = tmpPosition[i].y;
        double vx = tmpVelocity[i].x;
        double vy = tmpVelocity[i].y;

        int physicsUpdateIndex;
        for (physicsUpdateIndex = 0;
            physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
            ++physicsUpdateIndex)
        {
            double dx = x - tmpMousePosX;
            double dy = y - tmpMousePosY;
            double d2 = dx * dx + dy * dy;

            double invd = 1.0 / sqrt(d2);
            double invd2 = invd * invd;

            double ax = (GRAVITY * dx) * (invd * invd2);
            double ay = (GRAVITY * dy) * (invd * invd2);

            vx -= ax * dt;
            vy -= ay * dt;

            x += vx * dt;
            y += vy * dt;
        }

        // Single write-back per satellite
        tmpPosition[i].x = x;
        tmpPosition[i].y = y;
        tmpVelocity[i].x = vx;
        tmpVelocity[i].y = vy;
    }

    // Copy back into float storage once
    for (int idx2 = 0; idx2 < SATELLITE_COUNT; ++idx2) {
        satellites[idx2].position.x = (float)tmpPosition[idx2].x;
        satellites[idx2].position.y = (float)tmpPosition[idx2].y;
        satellites[idx2].velocity.x = (float)tmpVelocity[idx2].x;
        satellites[idx2].velocity.y = (float)tmpVelocity[idx2].y;
    }
}


// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine)
// Decides the color for each pixel.



////////////////////////////////////////////////
// ¤¤ DISABLED: PART1 GRAPHICSENGINE LOGIC ¤¤ //
////////////////////////////////////////////////

/* void parallelGraphicsEngine(void) {

    int tmpMousePosX = mousePosX;
    int tmpMousePosY = mousePosY;

    const float BH_R2 = BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS;
    const float SAT_R2 = SATELLITE_RADIUS * SATELLITE_RADIUS;

    int y;
#pragma omp parallel for schedule(static) // or: schedule(static, 2)
    for (y = 0; y < WINDOW_HEIGHT; ++y) {

        int idx = y * WINDOW_WIDTH;
        float py = (float)y;

        int x;
        for (x = 0; x < WINDOW_WIDTH; ++x, ++idx) {

            float px = (float)x;

            // Black hole test (no sqrt)
            float dxBH = px - tmpMousePosX;
            float dyBH = py - tmpMousePosY;
            float d2BH = dxBH * dxBH + dyBH * dyBH;
            if (d2BH < BH_R2) {
                pixels[idx].red = 0;
                pixels[idx].green = 0;
                pixels[idx].blue = 0;
                continue;
            }

            // Single-pass satellite loop
            float sumR = 0.f, sumG = 0.f, sumB = 0.f;
            float weights = 0.f;

            float shortestD2 = INFINITY;
            color_f32 nearestID = (color_f32){ 0.f, 0.f, 0.f };
            int hitsSatellite = 0;

            int j;
            for (j = 0; j < SATELLITE_COUNT; ++j) {

                float dx = px - satellites[j].position.x;
                float dy = py - satellites[j].position.y;
                float d2 = dx * dx + dy * dy;

                if (d2 < SAT_R2) {
                    pixels[idx].red = 255;
                    pixels[idx].green = 255;
                    pixels[idx].blue = 255;
                    hitsSatellite = 1;
                    break;
                }

                float w = 1.0f / (d2 * d2);
                weights += w;

                sumR += satellites[j].identifier.red * w;
                sumG += satellites[j].identifier.green * w;
                sumB += satellites[j].identifier.blue * w;

                if (d2 < shortestD2) {
                    shortestD2 = d2;
                    nearestID = satellites[j].identifier;
                }
            }

            if (!hitsSatellite) {
                float invW = 1.0f / weights;
                float r = nearestID.red + 3.0f * (sumR * invW);
                float g = nearestID.green + 3.0f * (sumG * invW);
                float b = nearestID.blue + 3.0f * (sumB * invW);

                pixels[idx].red = (uint8_t)(r * 255.0f);
                pixels[idx].green = (uint8_t)(g * 255.0f);
                pixels[idx].blue = (uint8_t)(b * 255.0f);
            }
        }
    }
}
*/



void parallelGraphicsEngine(void) {

    // prepare host SoA arrays each frame
    float h_pos_x[SATELLITE_COUNT];
    float h_pos_y[SATELLITE_COUNT];

    for (int j = 0; j < SATELLITE_COUNT; ++j) {
        h_pos_x[j] = satellites[j].position.x;
        h_pos_y[j] = satellites[j].position.y;
    }

    // write satellites to device
    CL_CHECK(clEnqueueWriteBuffer(clQ, d_pos_x, CL_FALSE, 0, sizeof(h_pos_x), h_pos_x, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(clQ, d_pos_y, CL_FALSE, 0, sizeof(h_pos_y), h_pos_y, 0, NULL, NULL));

    // locals (not macros) so we can take addresses safely
    float bh_r2 = BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS;
    float sat_r2 = SATELLITE_RADIUS * SATELLITE_RADIUS;
    int   mx = mousePosX;
    int   my = mousePosY;
    int   satCount = SATELLITE_COUNT;
    int   width = WINDOW_WIDTH;
    int   height = WINDOW_HEIGHT;

    // set kernel args
    int arg = 0;
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_pixels));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_pos_x));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_pos_y));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_id_r));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_id_g));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(cl_mem), &d_id_b));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(satCount), &satCount));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(width), &width));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(height), &height));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(bh_r2), &bh_r2));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(sat_r2), &sat_r2));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(mx), &mx));
    CL_CHECK(clSetKernelArg(clKer, arg++, sizeof(my), &my));

    // global dims rounded up to multiples of WG
    size_t local[2] = { WGX, WGY };
    size_t g0 = ((size_t)WINDOW_WIDTH + WGX - 1) / WGX * WGX;
    size_t g1 = ((size_t)WINDOW_HEIGHT + WGY - 1) / WGY * WGY;
    size_t global[2] = { g0, g1 };

    // launch
    CL_CHECK(clEnqueueNDRangeKernel(clQ, clKer, 2, NULL, global, local, 0, NULL, NULL));
    CL_CHECK(clFinish(clQ));

    CL_CHECK(clEnqueueReadBuffer(clQ, d_pixels, CL_TRUE, 0,
        sizeof(unsigned char) * 4 * SIZE, pixels, 0, NULL, NULL));
}




// ## You may add your own destrcution routines here ##
void destroy() {
    if (d_pixels) clReleaseMemObject(d_pixels);
    if (d_pos_x)  clReleaseMemObject(d_pos_x);
    if (d_pos_y)  clReleaseMemObject(d_pos_y);
    if (d_id_r)   clReleaseMemObject(d_id_r);
    if (d_id_g)   clReleaseMemObject(d_id_g);
    if (d_id_b)   clReleaseMemObject(d_id_b);
    if (clKer)    clReleaseKernel(clKer);
    if (clProg)   clReleaseProgram(clProg);
    if (clQ)      clReleaseCommandQueue(clQ);
    if (clCtx)    clReleaseContext(clCtx);
}



////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
SDL_Window* win;
SDL_Surface* surf;
// Is used to find out frame times
int totalTimeAcc, satelliteMovementAcc, pixelColoringAcc, frameCount;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine(){
    // Graphics pixel loop
    for(int i = 0 ;i < SIZE; ++i) {

      // Row wise ordering
      floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // Draw the black hole
      floatvector positionToBlackHole = {.x = pixel.x -
         HORIZONTAL_CENTER, .y = pixel.y - VERTICAL_CENTER};
      float distToBlackHoleSquared =
         positionToBlackHole.x * positionToBlackHole.x +
         positionToBlackHole.y * positionToBlackHole.y;
      float distToBlackHole = sqrt(distToBlackHoleSquared);
      if (distToBlackHole < BLACK_HOLE_RADIUS) {
         correctPixels[i].red = 0;
         correctPixels[i].green = 0;
         correctPixels[i].blue = 0;
         continue; // Black hole drawing done
      }

      // This color is used for coloring the pixel
      color_f32 renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

      // Find closest satellite
      float shortestDistance = INFINITY;

      float weights = 0.f;
      int hitsSatellite = 0;

      // First Graphics satellite loop: Find the closest satellite.
      for(int j = 0; j < SATELLITE_COUNT; ++j){
         floatvector difference = {.x = pixel.x - satellites[j].position.x,
                                   .y = pixel.y - satellites[j].position.y};
         float distance = sqrt(difference.x * difference.x +
                               difference.y * difference.y);

         if(distance < SATELLITE_RADIUS) {
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
            hitsSatellite = 1;
            break;
         } else {
            float weight = 1.0f / (distance*distance*distance*distance);
            weights += weight;
            if(distance < shortestDistance){
               shortestDistance = distance;
               renderColor = satellites[j].identifier;
            }
         }
      }

      // Second graphics loop: Calculate the color based on distance to every satellite.
      if (!hitsSatellite) {
         for(int j = 0; j < SATELLITE_COUNT; ++j){
            floatvector difference = {.x = pixel.x - satellites[j].position.x,
                                      .y = pixel.y - satellites[j].position.y};
            float dist2 = (difference.x * difference.x +
                           difference.y * difference.y);
            float weight = 1.0f/(dist2* dist2);

            renderColor.red += (satellites[j].identifier.red *
                                weight /weights) * 3.0f;

            renderColor.green += (satellites[j].identifier.green *
                                  weight / weights) * 3.0f;

            renderColor.blue += (satellites[j].identifier.blue *
                                 weight / weights) * 3.0f;
         }
      }
      correctPixels[i].red = (uint8_t) (renderColor.red * 255.0f);
      correctPixels[i].green = (uint8_t) (renderColor.green * 255.0f);
      correctPixels[i].blue = (uint8_t) (renderColor.blue * 255.0f);
    }
}

void sequentialPhysicsEngine(satellite *s){

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   doublevector tmpPosition[SATELLITE_COUNT];
   doublevector tmpVelocity[SATELLITE_COUNT];

   for (int i = 0; i < SATELLITE_COUNT; ++i) {
       tmpPosition[i].x = s[i].position.x;
       tmpPosition[i].y = s[i].position.y;
       tmpVelocity[i].x = s[i].velocity.x;
       tmpVelocity[i].y = s[i].velocity.y;
   }

   // Physics iteration loop
   for(int physicsUpdateIndex = 0;
       physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex){

       // Physics satellite loop
      for(int i = 0; i < SATELLITE_COUNT; ++i){

         // Distance to the blackhole
         // (bit ugly code because C-struct cannot have member functions)
         doublevector positionToBlackHole = {.x = tmpPosition[i].x -
            HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER};
         double distToBlackHoleSquared =
            positionToBlackHole.x * positionToBlackHole.x +
            positionToBlackHole.y * positionToBlackHole.y;
         double distToBlackHole = sqrt(distToBlackHoleSquared);

         // Gravity force
         doublevector normalizedDirection = {
            .x = positionToBlackHole.x / distToBlackHole,
            .y = positionToBlackHole.y / distToBlackHole};
         double accumulation = GRAVITY / distToBlackHoleSquared;

         // Delta time is used to make velocity same despite different FPS
         // Update velocity based on force
         tmpVelocity[i].x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpVelocity[i].y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

         // Update position based on velocity
         tmpPosition[i].x +=
            tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpPosition[i].y +=
            tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
      }
   }

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   // copy back the float storage.
   for (int i = 0; i < SATELLITE_COUNT; ++i) {
       s[i].position.x = tmpPosition[i].x;
       s[i].position.y = tmpPosition[i].y;
       s[i].velocity.x = tmpVelocity[i].x;
       s[i].velocity.y = tmpVelocity[i].y;
   }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_ERROR 10
#define ALLOWED_NUMBER_OF_ERRORS 10
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck(){
   int countErrors = 0;
   for(unsigned int i=0; i < SIZE; ++i) {
      if(abs(correctPixels[i].red - pixels[i].red) > ALLOWED_ERROR ||
         abs(correctPixels[i].green - pixels[i].green) > ALLOWED_ERROR ||
         abs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_ERROR) {
         printf("Pixel x=%d y=%d value: %d, %d, %d. Should have been: %d, %d, %d\n",
                i % WINDOW_WIDTH, i / WINDOW_WIDTH,
                pixels[i].red, pixels[i].green, pixels[i].blue,
                correctPixels[i].red, correctPixels[i].green, correctPixels[i].blue);
         countErrors++;
         if (countErrors > ALLOWED_NUMBER_OF_ERRORS) {
            printf("Too many errors (%d) in frame %d, Press enter to continue.\n", countErrors, frameNumber);
            getchar();
            return;
         }
       }
   }
   printf("Error check passed with acceptable number of wrong pixels: %d\n", countErrors);
}


// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void){
   int timeSinceStart = SDL_GetTicks();

   // Error check during first frames
   if (frameNumber < 2) {
      memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
      sequentialPhysicsEngine(backupSatelites);
      mousePosX = HORIZONTAL_CENTER;
      mousePosY = VERTICAL_CENTER;
   } else {
      SDL_GetMouseState(&mousePosX, &mousePosY);
      if ((mousePosX == 0) && (mousePosY == 0)) {
         mousePosX = HORIZONTAL_CENTER;
         mousePosY = VERTICAL_CENTER;
      }
   }
   parallelPhysicsEngine();
   if (frameNumber < 2) {
      for (int i = 0; i < SATELLITE_COUNT; i++) {
         if (memcmp (&satellites[i], &backupSatelites[i], sizeof(satellite))) {
            printf("Incorrect satellite data of satellite: %d\n", i);
            getchar();
         }
      }
   }

   int satelliteMovementMoment = SDL_GetTicks();
   int satelliteMovementTime = satelliteMovementMoment  - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   int pixelColoringMoment = SDL_GetTicks();
   int pixelColoringTime =  pixelColoringMoment - satelliteMovementMoment;

   int finishTime = SDL_GetTicks();
   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
      errorCheck();
   } else if (frameNumber == 2) {
      previousFinishTime = finishTime;
      printf("Time spent on moving satellites + Time spent on space coloring : Total time in milliseconds between frames (might not equal the sum of the left-hand expression)\n");
   } else if (frameNumber > 2) {
     // Print timings
     int totalTime = finishTime - previousFinishTime;
     previousFinishTime = finishTime;

     printf("Latency of this frame %i + %i : %ims \n",
             satelliteMovementTime, pixelColoringTime, totalTime);

     frameCount++;
     totalTimeAcc += totalTime;
     satelliteMovementAcc += satelliteMovementTime;
     pixelColoringAcc += pixelColoringTime;
     printf("Averaged over all frames: %i + %i : %ims.\n",
             satelliteMovementAcc/frameCount, pixelColoringAcc/frameCount, totalTimeAcc/frameCount);

   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max){
   return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed){

   if(seed != 0){
     srand(seed);
   }

   // Init pixel buffer which is rendered to the widow
   pixels = (color_u8*)malloc(sizeof(color_u8) * SIZE);

   // Init pixel buffer which is used for error checking
   correctPixels = (color_u8*)malloc(sizeof(color_u8) * SIZE);

   backupSatelites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);


   // Init satellites buffer which are moving in the space
   satellites = (satellite*)malloc(sizeof(satellite) * SATELLITE_COUNT);

   // Create random satellites
   for(int i = 0; i < SATELLITE_COUNT; ++i){

      // Random reddish color
      color_f32 id = {.red = randomNumber(0.f, 0.15f) + 0.1f,
                  .green = randomNumber(0.f, 0.14f) + 0.0f,
                  .blue = randomNumber(0.f, 0.16f) + 0.0f};

      // Random position with margins to borders
      floatvector initialPosition = {.x = HORIZONTAL_CENTER - randomNumber(50, 320),
                              .y = VERTICAL_CENTER - randomNumber(50, 320) };
      initialPosition.x = (i / 2 % 2 == 0) ?
         initialPosition.x : WINDOW_WIDTH - initialPosition.x;
      initialPosition.y = (i < SATELLITE_COUNT / 2) ?
         initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

      // Randomize velocity tangential to the balck hole
      floatvector positionToBlackHole = {.x = initialPosition.x - HORIZONTAL_CENTER,
                                    .y = initialPosition.y - VERTICAL_CENTER};
      float distance = (0.06 + randomNumber(-0.01f, 0.01f))/
        sqrt(positionToBlackHole.x * positionToBlackHole.x +
          positionToBlackHole.y * positionToBlackHole.y);
      floatvector initialVelocity = {.x = distance * -positionToBlackHole.y,
                                .y = distance * positionToBlackHole.x};

      // Every other orbits clockwise
      if(i % 2 == 0){
         initialVelocity.x = -initialVelocity.x;
         initialVelocity.y = -initialVelocity.y;
      }

      satellite tmpSatelite = {.identifier = id, .position = initialPosition,
                              .velocity = initialVelocity};
      satellites[i] = tmpSatelite;
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void){
   destroy();

   free(pixels);
   free(correctPixels);
   free(satellites);

   if(seed != 0){
     printf("Used seed: %i\n", seed);
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window
void render(void){
   SDL_LockSurface(surf);
   memcpy(surf->pixels, pixels, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
   SDL_UnlockSurface(surf);

   SDL_UpdateWindowSurface(win);
   frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits render window and starts mainloop
int main(int argc, char** argv){

   if(argc > 1){
     seed = atoi(argv[1]);
     printf("Using seed: %i\n", seed);
   }

   SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);
   win = SDL_CreateWindow(
        "Satellites",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        0
    );
   surf = SDL_GetWindowSurface(win);

   fixedInit(seed);
   init();

   SDL_Event event;
   int running = 1;
   while (running) {
      while (SDL_PollEvent(&event)) switch (event.type) {
         case SDL_QUIT:
            printf("Quit called\n");
            running = 0;
            break;
      }
      compute();
      render();
   }
   SDL_Quit();
   fixedDestroy();
}
