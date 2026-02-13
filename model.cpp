#include "model.hpp"
#include <cstdio>
#define MEGA 1.0e-6
#define GIGA 1.0e-9
#define WRITE_DATA 0

uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

void ReportProblemSizeCSV(const int sx, const int sy, const int sz,
			  const int bord, const int st,
			  FILE *f){
  fprintf(f,
	  "sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",
	  sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,
		      long HWM, char *HWMUnit, FILE *f){
  fprintf(f,
	  "walltime; %lf; GSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",
	  walltime, MSamples, HWM, HWMUnit);
}

void Model(const int st, const int iSource, const float dtOutput,
    const int nx,   const int ny, const int nz,
    const int sx,   const int sy, const int sz,
    const float dx, const float dy, const float dz,
    const float dt, const int it,
    const int bord, const int absorb,
    SlicePtr sPtr,
    int argc, char **argv)
{
    float tSim=0.0;
    int   nOut=1;
    float tOut=nOut*dtOutput;

    const long samplesPropagate=(long)(sx-2*bord)*(long)(sy-2*bord)*(long)(sz-2*bord);
    const long totalSamples=samplesPropagate*(long)st;

    double walltime=0.0;
    uint64_t stamp1 = get_timestamp_ns();

    std::unique_ptr<Driver> driver = createDriver(argc, argv);
    printf("Initializing driver...\n");
    driver->initialize(sx, sy, sz, dx, dy, dz, dt, nx, ny, nz, bord, absorb);

    for (int it=1; it<=st; it++) {

        // Calculate / obtain source value on i timestep
        float src_value = Source(dt, it-1);
        driver->insertSource(iSource, src_value);

        const double t0=wtime();
        printf("propagate\n");
        driver->propagate();
        walltime+=wtime()-t0;

        tSim=it*dt;
        if (tSim >= tOut) {
            if(WRITE_DATA){
                printf("update host\n");
                driver->updateHost();
                DumpSliceFile_Nofor(sx,sy,sz,driver->getData(), sPtr);
            }
            tOut=(++nOut)*dtOutput;
        }
  }

  // close binary output file before measuring time to include total io time
  uint64_t stamp2 = get_timestamp_ns();

  // // get HWM data
  const char StringHWM[6]="VmHWM";
  char line[256], HWMUnit[8];
  long HWM;
  const double MSamples=(GIGA*(double)totalSamples)/walltime;

  FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);

  // Dump Execution Metrics
  double execution_time = ((double)(stamp2-stamp1))*1e-9;

  // printf("Total dump time (s): %f\n", tdt);
  printf ("Execution time (s) is %lf\n", walltime);
  printf ("Total execution time (s) is %lf\n", execution_time);
  printf ("GSamples/s %.0lf\n", MSamples);
  printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  printf("%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%f,%f,%lu,%lu,%lf,%lf,%.2f\n",
          BACKEND, sx - 2*bord - 2*absorb, sy - 2*bord - 2*absorb, sz - 2*bord - 2*absorb, absorb, dx, dy, dz, dt, st*dt,
          stamp1, stamp2, walltime, execution_time, MSamples);

  // Dump Execution Metrics in CSV
  FILE *fr=NULL;
  const char fName[]="Report.csv";
  fr=fopen(fName,"w");

  // report problem size
  ReportProblemSizeCSV(sx, sy, sz,
		       bord, st,
		       fr);

  // report collected metrics
  ReportMetricsCSV(walltime, MSamples,
		   HWM, HWMUnit, fr);

  fclose(fr);
  fflush(stdout);
  driver->finalize();
}
