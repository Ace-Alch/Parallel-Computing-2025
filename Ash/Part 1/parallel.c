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








// ## You may add your own variables here ##




// ## You may add your own initialization routines here ##
void init(){


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
void parallelGraphicsEngine(void) {

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


// ## You may add your own destrcution routines here ##
void destroy(){


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
