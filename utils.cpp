#include "utils.hpp"

// OpenSliceFile: open file in RFS format that will be continuously appended
SlicePtr OpenSliceFile(int ixStart, int ixEnd,
		       int iyStart, int iyEnd,
		       int izStart, int izEnd,
		       float dx, float dy, float dz, float dt,
		       char *fName)
{
    //PPL  char procName[128]="**(OpenSliceFile)**";
    SlicePtr ret;
    ret = (SlicePtr) malloc(sizeof(Slice));

    // verify slice direction
    if (ixStart==ixEnd)
        ret->direction=XSLICE;
    else if (iyStart==iyEnd)
        ret->direction=YSLICE;
    else if (izStart==izEnd)
        ret->direction=ZSLICE;
    else {
        ret->direction=FULL;
    }

    // header and binary file names
    strcpy(ret->fName,fName);
    strcpy(ret->fNameHeader,fName);
    strcat(ret->fNameHeader,".rsf");
    strcpy(ret->fNameBinary,FNAMEBINARYPATH);
    strcat(ret->fNameBinary,ret->fNameHeader);
    strcat(ret->fNameBinary,"@");

    // create header and binary files in rsf format
    ret->fpHead=fopen(ret->fNameHeader, "w+");
    ret->fpBinary=fopen(ret->fNameBinary, "w+");
    ret->ixStart=ixStart;
    ret->ixEnd=ixEnd;
    ret->iyStart=iyStart;
    ret->iyEnd=iyEnd;
    ret->izStart=izStart;
    ret->izEnd=izEnd;
    ret->itCnt=0;
    ret->dx=dx;
    ret->dy=dy;
    ret->dz=dz;
    ret->dt=dt;

    char sName[16];
    switch(ret->direction) {
        case XSLICE:
            strcpy(sName,"XSlice");
            break;
        case YSLICE:
            strcpy(sName,"YSlice");
            break;
        case ZSLICE:
            strcpy(sName,"ZSlice");
            break;
        case FULL:
            strcpy(sName,"Grid Section");
            break;
    }
    return(ret);
}

void DumpSliceFile_Nofor(int sx, int sy, int sz,
		   float *arrP, SlicePtr p)
{
    //PPL  int ix, iy, iz;
    int totalSize = sx * sy * sz;

    // dump section to binary file
    fwrite((void *) arrP,
    sizeof(float),
    totalSize,
    p->fpBinary);

    // increase it count
    p->itCnt++;
}

void CloseSliceFile(SlicePtr p)
{
    fprintf(p->fpHead,"in=\"%s\"\n", p->fNameBinary);
    fprintf(p->fpHead,"data_format=\"native_float\"\n");
    fprintf(p->fpHead,"esize=%lu\n", sizeof(float));
    switch(p->direction) {
        case XSLICE:
            fprintf(p->fpHead,"n1=%d\n",p->iyEnd-p->iyStart+1);
            fprintf(p->fpHead,"n2=%d\n",p->izEnd-p->izStart+1);
            fprintf(p->fpHead,"n3=%d\n",p->itCnt);
            fprintf(p->fpHead,"d1=%f\n",p->dy);
            fprintf(p->fpHead,"d2=%f\n",p->dz);
            fprintf(p->fpHead,"d3=%f\n",p->dt);
            break;
        case YSLICE:
            fprintf(p->fpHead,"n1=%d\n",p->ixEnd-p->ixStart+1);
            fprintf(p->fpHead,"n2=%d\n",p->izEnd-p->izStart+1);
            fprintf(p->fpHead,"n3=%d\n",p->itCnt);
            fprintf(p->fpHead,"d1=%f\n",p->dx);
            fprintf(p->fpHead,"d2=%f\n",p->dz);
            fprintf(p->fpHead,"d3=%f\n",p->dt);
            break;
        case ZSLICE:
            fprintf(p->fpHead,"n1=%d\n",p->ixEnd-p->ixStart+1);
            fprintf(p->fpHead,"n2=%d\n",p->iyEnd-p->iyStart+1);
            fprintf(p->fpHead,"n3=%d\n",p->itCnt);
            fprintf(p->fpHead,"d1=%f\n",p->dx);
            fprintf(p->fpHead,"d2=%f\n",p->dy);
            fprintf(p->fpHead,"d3=%f\n",p->dt);
            break;
        case FULL:
            fprintf(p->fpHead,"n1=%d\n",p->ixEnd-p->ixStart+1);
            fprintf(p->fpHead,"n2=%d\n",p->iyEnd-p->iyStart+1);
            fprintf(p->fpHead,"n3=%d\n",p->izEnd-p->izStart+1);
            fprintf(p->fpHead,"n4=%d\n",p->itCnt);
            fprintf(p->fpHead,"d1=%f\n",p->dx);
            fprintf(p->fpHead,"d2=%f\n",p->dy);
            fprintf(p->fpHead,"d3=%f\n",p->dz);
            fprintf(p->fpHead,"d4=%f\n",p->dt);
            break;
    }
    fclose(p->fpHead);
    fclose(p->fpBinary);
}
