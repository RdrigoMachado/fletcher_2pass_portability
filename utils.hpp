#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define FNAMEBINARYPATH "./"


enum sliceDirection {XSLICE, YSLICE, ZSLICE, FULL};


typedef struct tsection {
  enum sliceDirection direction;
  int ixStart;
  int ixEnd;
  int iyStart;
  int iyEnd;
  int izStart;
  int izEnd;
  int itCnt;
  float dx;
  float dy;
  float dz;
  float dt;
  FILE *fpHead;
  FILE *fpBinary;
  char fName[128];
  char fNameHeader[128];
  char fNameBinary[128];
} Slice, *SlicePtr;

SlicePtr OpenSliceFile(int ixStart, int ixEnd,
		       int iyStart, int iyEnd,
		       int izStart, int izEnd,
		       float dx, float dy, float dz, float dt,
		       char *fName);

void DumpSliceFile_Nofor(int sx, int sy, int sz,
		   float *arrP, SlicePtr p);

void CloseSliceFile(SlicePtr p);
