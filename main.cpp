#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include "model.hpp"
#include <string.h>
#include <math.h>
#include "utils.hpp"

#define BORD 4
#define FLETCHER_ARGS 11
#define indirecao(ix,iy,iz) (((iz)*sy+(iy))*sx+(ix))

int main(int argc, char** argv){

    int nx;                // grid points in x
    int ny;                // grid points in y
    int nz;                // grid points in z
    int bord=BORD;            // border size to apply the stencil at grid extremes
    int absorb;            // absortion zone size
    int sx;                // grid dimension in x (grid points + 2*border + 2*absortion)
    int sy;                // grid dimension in y (grid points + 2*border + 2*absortion)
    int sz;                // grid dimension in z (grid points + 2*border + 2*absortion)
    int st;                // number of time steps
    float dx;              // grid step in x
    float dy;              // grid step in y
    float dz;              // grid step in z
    float dt;              // time advance at each time step
    float tmax;            // desired simulation final time
    int ixSource;          // source x index
    int iySource;          // source y index
    int izSource;          // source z index
    int iSource;           // source index (ix,iy,iz) maped into 1D array
    int it = 0;            // for indices
    char fNameSec[128];    // prefix of sections files
    const float dtOutput=0.01;


    if (argc < FLETCHER_ARGS) {
        printf("program requires %d input arguments; execution halted\n",FLETCHER_ARGS-1);
        exit(-1);
    }

    strcpy(fNameSec,argv[1]);
    nx=atoi(argv[2]);
    ny=atoi(argv[3]);
    nz=atoi(argv[4]);
    absorb=atoi(argv[5]);
    dx=atof(argv[6]);
    dy=atof(argv[7]);
    dz=atof(argv[8]);
    dt=atof(argv[9]);
    tmax=atof(argv[10]);
    // grid dimensions from problem size
    sx=nx+2*bord+2*absorb;
    sy=ny+2*bord+2*absorb;
    sz=nz+2*bord+2*absorb;
    // number of time iterations
    st=ceil(tmax/dt);
    // source position
    ixSource=sx/2;
    iySource=sy/2;
    izSource=sz/2;
    iSource=indirecao(ixSource,iySource,izSource);

    int ixStart=0;
    int ixEnd=sx-1;
    int iyStart=0;
    int iyEnd=sy-1;
    int izStart=0;
    int izEnd=sz-1;

    SlicePtr sPtr;
    sPtr=OpenSliceFile(ixStart, ixEnd,
		     iyStart, iyEnd,
		     izStart, izEnd,
		     dx, dy, dz, dt,
		     fNameSec);

    Model(st, iSource, dtOutput, nx, ny, nz, sx, sy, sz, dx, dy, dz, dt, it, bord, absorb, sPtr, argc, argv);

    CloseSliceFile(sPtr);
    return 0;
}
