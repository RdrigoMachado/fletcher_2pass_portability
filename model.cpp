#include "model.hpp"
#include <cstdio>
#include <vector>
#include <string>

#define MEGA 1.0e-6
#define GIGA 1.0e-9
#define WRITE_DATA 1  // Set to 1 to enable file writing

uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

// Simple timer helper
double wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
    float tSim = 0.0;
    int   nOut = 1;
    float tOut = nOut * dtOutput;

    // --- Metrics Variables ---
    double t_initialize = 0.0;
    double t_finalize   = 0.0;
    double t_propagate  = 0.0;
    double t_insert     = 0.0;
    double t_updateHost = 0.0;
    double t_loop_total = 0.0; // Measures the total walltime of the main loop

    // --- Driver Creation ---
    std::unique_ptr<Driver> driver = createDriver(argc, argv);

    // --- Initialization Phase ---
    printf("Initializing driver...\n");
    double t0 = wtime();
    driver->initialize(sx, sy, sz, dx, dy, dz, dt, nx, ny, nz, bord, absorb);
    t_initialize = wtime() - t0;

    // --- Simulation Loop ---
    double loop_start = wtime();

    for (int it = 1; it <= st; it++) {

        // 1. Insert Source
        t0 = wtime();
        float src_value = Source(dt, it - 1);
        driver->insertSource(iSource, src_value);
        t_insert += (wtime() - t0);

        // 2. Propagate
        t0 = wtime();
        // printf("propagate\n"); // Commented out to avoid I/O noise in timing
        driver->propagate();
        t_propagate += (wtime() - t0);

        tSim = it * dt;

        // 3. Output / Update Host
        if (tSim >= tOut) {

            // Always update host when output time is reached (as requested)
            t0 = wtime();
            // printf("update host\n");
            driver->updateHost();
            t_updateHost += (wtime() - t0);

            // Only dump to file if enabled
            if (WRITE_DATA) {
                DumpSliceFile_Nofor(sx, sy, sz, driver->getData(), sPtr);
            }

            tOut = (++nOut) * dtOutput;
        }
    }

    t_loop_total = wtime() - loop_start;

    // --- Memory Metrics (High Water Mark) ---
    const char StringHWM[6] = "VmHWM";
    char line[256], HWMUnit[8];
    long HWM = 0;

    FILE *fp = fopen("/proc/self/status", "r");
    if (fp) {
        while (fgets(line, 256, fp) != NULL) {
            if (strncmp(line, StringHWM, 5) == 0) {
                sscanf(line + 6, "%ld %s", &HWM, HWMUnit);
                break;
            }
        }
        fclose(fp);
    }

    // --- Finalize Phase ---
    t0 = wtime();
    driver->finalize();
    t_finalize = wtime() - t0;

    // --- Console Report ---
    const long samplesPropagate = (long)(sx - 2 * bord) * (long)(sy - 2 * bord) * (long)(sz - 2 * bord);
    const long totalSamples = samplesPropagate * (long)st;
    const double GSamples = (GIGA * (double)totalSamples) / t_propagate; // GSamples based on pure propagation time usually

    // printf("==========================================\n");
    // printf("Backend: %s\n", BACKEND);
    // printf("Grid: %d x %d x %d (Steps: %d)\n", sx, sy, sz, st);
    // printf("------------------------------------------\n");
    // printf("Initialization : %.6f s\n", t_initialize);
    // printf("Insert Source  : %.6f s\n", t_insert);
    // printf("Propagate      : %.6f s\n", t_propagate);
    // printf("Update Host    : %.6f s\n", t_updateHost);
    // printf("Finalize       : %.6f s\n", t_finalize);
    // printf("------------------------------------------\n");
    // printf("Total Loop Time: %.6f s\n", t_loop_total);
    // printf("GSamples/s     : %.2f\n", MSamples);
    // printf("Memory HWM     : %ld %s\n", HWM, HWMUnit);
    // printf("==========================================\n");

    // --- CSV Output (Requested Format) ---
    // Format: version,sx,sy,sz,dt,st,walltime,propagate,insertsource,updatehost,initialize,finalize,gsamples
    printf("%s,%d,%d,%d,%f,%d,%f,%f,%f,%f,%f,%f,%.2f\n",
           BACKEND,
           sx, sy, sz,
           dt, st,
           t_loop_total,
           t_propagate,
           t_insert,
           t_updateHost,
           t_initialize,
           t_finalize,
           GSamples
    );

    fflush(stdout);
}
