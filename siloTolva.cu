#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "estructuras.h"

#define DIAMS 1.2
#define THS_MAX 256
#define MAX_LOTS 3

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if(cudaSuccess != err)
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif
	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void clean(double3*, double3*, grain_prop*, long);
__global__ void cleanTouch(touch*, long);
__global__ void cleanProf(double*, double*, long);
__global__ void cellLocate(double3*, long*, long*, parameters);
__global__ void getContacts(double3*, grain_prop*, long*, long*, long*,
		long*, double3*, parameters);
__global__ void getForces(double3*, double3*, double3*, double3*, double3*,
		grain_prop*, touch*, long*, long*, double3*, double3*,
		double3*, parameters);
__global__ void verletInit(double3*, double3*, double3*, double3*, double3*,
		grain_prop*, parameters, int*);
__global__ void verletFinish(double3*, double3*, double3*, double3*, double3*,
		double3*, grain_prop*, parameters);
__global__ void getPhi(double3*, grain_prop*, long*, long*, long*,
		parameters);
__global__ void getVVprof(double3*, double3*, int*, double*,
		double*, long, double, double);

// Uniform random number generator between 0.0 and 1.0
double rannew64(long*);

typedef struct
{
	int lot_id;
	long count;
	double spawn_time;
	double3 spawn_pos;
} lot_cfg;

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

void xyzOvPrint(FILE *fSnap, grain_prop *grainVec, double3 *rrVec, double radMin,
		double radMax, parameters pars)
{
	int molType, partType;
	static int flag_init = 1;
	long mm;
	static long ngrains, npart, *exGrain;
	static double siloInit, siloWidth, siloHeight, siloThick,
		hopWidth, hopH, boxDim;
	double rad;
	double3 rr;

	if (flag_init)
	{
		ngrains = pars.ngrains;
		siloInit = pars.siloInit;
		siloWidth = pars.siloWidth;
		siloHeight = pars.siloHeight;
		siloThick = pars.siloThick;
		boxDim = siloInit + siloHeight;
		hopWidth = pars.hopWidth;
		hopH = pars.bottGap;
		npart = ngrains;
		exGrain = (long *) malloc(ngrains*sizeof(long));
		memset(exGrain, 1, ngrains*sizeof(long));
		flag_init = 0;
	}

	// Print in .xyz format for Ovito
	fprintf(fSnap, "%ld\n", npart + 6);
	fprintf(fSnap, "Lattice=");
	fprintf(fSnap, "\"%.2f 0.00 0.00 ", siloWidth);
	fprintf(fSnap, "0.00 %.2f 0.00 ", siloThick);
	fprintf(fSnap, "0.00 0.00 %.2f\"\n", boxDim);

	// Print vertices
	molType = 0;
	partType = 0;
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		0.0, 0.5*siloThick, boxDim, molType, partType);
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		0.0, 0.5*siloThick, siloInit, molType, partType);
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		0.5*(siloWidth - hopWidth), 0.5*siloThick, hopH, molType, partType);
	molType = 1;
	partType = 0;
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		siloWidth, 0.5*siloThick, boxDim, molType, partType);
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		siloWidth, 0.5*siloThick, siloInit, molType, partType);
	fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n", 0.2*radMin,
		0.5*(siloWidth + hopWidth), 0.5*siloThick, hopH, molType, partType);

	// Print grains
	molType = 2;
	for (mm=0; mm<ngrains; mm++)
	{
		if (!exGrain[mm]) continue;

		rr = rrVec[mm];
		if (rr.z < 0.0)
		{
			npart--;
			exGrain[mm] = 0;
		}

		rad = grainVec[mm].rad;
		if (rad - radMin <= 0.5*(radMax - radMin)) partType = 1;
		else partType = 2;

		fprintf(fSnap, "%f\t%f\t%f\t%f\t%d\t%d\n",
		rad, rr.x, rr.y + 0.5*siloThick, rr.z, molType, partType);
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

int dataPrint(grain_prop *grainVec, double3 *rrVec, double3 *vvVec, double3 *wwVec,
	long *nCntcVec, long *tagCntcVec, double3 *rrcCntcVec, double3 *ffcCntcVec,
	parameters pars, long frame)
{
	double3 rr, vv, ww, rrc, ffc;
	long ii, jj, npart, ngrains, nTouch,
		indCntc, indCntc_init, indCntc_end;
	double rad, mass, bottGap, winWidth, siloWidth;
	char dir[100];
	FILE *fData, *fCntc;

	sprintf(dir, "DataFrames/grainsData%ld.dat", frame);
	fData = fopen(dir, "w");
	fprintf(fData, "# id_i\trad\tmasa\trrx\trry\trrz\tvvx\tvvy\tvvz\t"
			"wwx\twwy\twwz\n");

	sprintf(dir, "DataFrames/contactData%ld.dat", frame);
	fCntc = fopen(dir, "w");
	fprintf(fCntc, "# id_i\tid_j\trrijx\trrijy\trrijz\tffijx\tffijy\tffijz\n");

	ngrains = pars.ngrains;
	nTouch = pars.nTouch;
	bottGap = pars.bottGap;
	winWidth = pars.winWidth;
	siloWidth = pars.siloWidth;

	// Print grains
	npart = 0;
	for (ii=0; ii<ngrains; ii++)
	{
		rr = rrVec[ii];
		if (rr.z <= 0.0) continue;
		if (rr.z >= bottGap + winWidth) continue;
		if (rr.x <= 0.5*(siloWidth - winWidth)) continue;
		if (rr.x >= 0.5*(siloWidth + winWidth)) continue;

		rad = grainVec[ii].rad;
		mass = grainVec[ii].mass;

		vv = vvVec[ii];
		ww = wwVec[ii];

		fprintf(fData, "%ld\t%lf\t%lf\t%lf\t%lf\t%lf\t"
				"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
				ii, rad, mass, rr.x, rr.y, rr.z,
				vv.x, vv.y, vv.z, ww.x, ww.y, ww.z);

		indCntc_init = ii*nTouch;
		indCntc_end = indCntc_init + nCntcVec[ii];

		for (indCntc=indCntc_init; indCntc<indCntc_end; indCntc++)
		{
			jj = tagCntcVec[indCntc];
			if (jj < 0) continue;
			if (ii >= jj) continue;

			rrc = rrcCntcVec[indCntc];
			ffc = ffcCntcVec[indCntc];

			fprintf(fCntc, "%ld\t%ld\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
				ii, jj, rrc.x, rrc.y, rrc.z, ffc.x, ffc.y, ffc.z);
		}

		npart++;
	}

	fclose(fData);
	fclose(fCntc);

	if (npart) return 0;
	else return 1;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

// Find the next power of two
long nextPow2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/*+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+ MAIN =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+*/

int main()
{
	/*+*+*+*+*+*+*+*+*+*+*+*+*+ PARÃMETROS +*+*+*+*+*+*+*+*+*+*+*+*+*/

	double siloWidth, siloHeight, siloThick, hopWidth, hopAng,
		bottGap, diamMin, diamMax, rho_g, rho_w, rho_p,
		tColl, eps_gg, mu_gg, eps_gw, mu_gw, eps_gp, mu_gp,
		xk_tn, xg_tn, xmu_ds, dt, gapFreq, tRun, tTrans, v0,
		winWidth, tapOpenTime;
	long ngrains, idum, nBinsHop;
	long dischargeTarget;
	int polyFlag, snapFlag, err_flag = 0;
	char renglon[200];
	struct stat dirStat;
	lot_cfg lots[MAX_LOTS];
	int lot_count = 0;
	long lots_total = 0;

	// Silo width; Silo height
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf %lf %lf", &siloWidth, &siloHeight, &siloThick);

	// Orifice width; Hopper angle; Orifice gap
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf %lf", &hopWidth, &bottGap);

	// Hopper angle
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf", &hopAng);

	// Number of grains
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &ngrains);

	// Polydispersity?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &polyFlag);

	// Grain radius
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &diamMin);

	// Radius difference
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &diamMax);

	// Density
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf %lf", &rho_g, &rho_w, &rho_p);

	// Collision time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tColl);

	// Epsilon and mu grain-grain
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gg, &mu_gg);

	// Epsilon and mu grain-wall
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gw, &mu_gw);

	// Epsilon and mu grain-plane
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gp, &mu_gp);

	// Ratio kappa_t/kappa_n; gamma_t/gamma_n; mu_d/mu_s
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf %lf", &xk_tn, &xg_tn, &xmu_ds);

	// Time step
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &dt);

	// Print gap in dt's
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &gapFreq);

	// Simulation time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tRun);

	// Transient time
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tTrans);

	// Initial velocity
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &v0);

	// Measurement window width
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &winWidth);

	// Print snapshots?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &snapFlag);

	// Random number seed
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &idum);

	// Number of bins at the orifice
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &nBinsHop);

	if (err_flag)
	{
		printf("Error en el archivo (.data) de parÃ¡metros.\n");
		exit (1);
	}

	{
		const char *lotsPath = "lots.data";
		FILE *fLots = fopen(lotsPath, "r");
		int header_read = 0;

		if (!fLots)
		{
			printf("Error: no se pudo abrir %s\n", lotsPath);
			exit (1);
		}

		while (fgets(renglon, sizeof(renglon), fLots) != NULL)
		{
			if (renglon[0] == '#' || renglon[0] == '\n' || renglon[0] == '\r')
				continue;

			if (!header_read)
			{
				if (sscanf(renglon, "%lf %ld", &tapOpenTime,
						&dischargeTarget) != 2)
				{
					printf("Error: formato invalido en %s\n", lotsPath);
					exit (1);
				}
				header_read = 1;
				continue;
			}

			if (lot_count >= MAX_LOTS)
			{
				printf("Error: demasiados lotes en %s (max %d)\n",
					lotsPath, MAX_LOTS);
				exit (1);
			}

			if (sscanf(renglon, "%d %ld %lf %lf %lf %lf",
					&lots[lot_count].lot_id,
					&lots[lot_count].count,
					&lots[lot_count].spawn_time,
					&lots[lot_count].spawn_pos.x,
					&lots[lot_count].spawn_pos.y,
					&lots[lot_count].spawn_pos.z) != 6)
			{
				printf("Error: formato invalido en %s\n", lotsPath);
				exit (1);
			}

			if (lots[lot_count].lot_id < 1 || lots[lot_count].lot_id > 3)
			{
				printf("Error: lot_id fuera de rango en %s\n", lotsPath);
				exit (1);
			}

			lots_total += lots[lot_count].count;
			lot_count++;
		}

		fclose(fLots);

		if (!header_read)
		{
			printf("Error: encabezado faltante en %s\n", lotsPath);
			exit (1);
		}

		if (lots_total != ngrains)
		{
			printf("Error: la suma de lotes (%ld) no coincide con ngrains (%ld)\n",
				lots_total, ngrains);
			exit (1);
		}
	}

	if (siloThick < diamMax)
	{
		printf("Error: El espesor del silo no puede ser menor"
			" que el diametro mÃ¡ximo de grano.\n");
		exit (2);
	}

	if (stat("DataFrames", &dirStat) == -1) mkdir("DataFrames", 0700);

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ PROPERTIES +*+*+*+*+*+*+*+*+*+*+*+*+*/

	grain_prop *grainVec;
	parameters pars;
	long mm;
	double radAve, massAve_g, massAve_w, massAve_p, mEff,
		aux_0, aux_1, gamma_gg, kappa_gg, gamma_gw,
		kappa_gw, gamma_gp, kappa_gp, deltaRad, random,
		rad_aux, massG,	totMass;
	double hopAngR, hopLength, radMin, radMax, siloInit;
	FILE *fBit;

	// Compute average masses
	radMin = diamMin/2.0;
	radMax = diamMax/2.0;
	deltaRad = (diamMax - diamMin)/2.0;
	radAve = radMin + 0.5*deltaRad;
	massAve_g = (4.0*PI/3.0)*radAve*radAve*radAve*rho_g;
	massAve_w = (4.0*PI/3.0)*radAve*radAve*radAve*rho_w;
	massAve_p = (4.0*PI/3.0)*radAve*radAve*radAve*rho_p;

	// Compute kappa and gamma Grain-Grain
	mEff = massAve_g/2.0;
	gamma_gg = -2.0*mEff*log(eps_gg)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gg)/tColl;
	kappa_gg = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Compute kappa and gamma Grain-Wall
	mEff = massAve_g*massAve_w/(massAve_g + massAve_w);
	gamma_gw = -2.0*mEff*log(eps_gw)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gw)/tColl;
	kappa_gw = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Compute kappa and gamma Grain-Plane
	mEff = massAve_g*massAve_p/(massAve_g + massAve_p);
	gamma_gp = -2.0*mEff*log(eps_gp)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gp)/tColl;
	kappa_gp = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Get CPU-GPU memory in UNIFIED MEMORY
	cudaSafeCall(cudaMallocManaged(&grainVec, ngrains*sizeof(grain_prop)));

	// Compute radii, masses, inertias and store
	for (mm=0; mm<ngrains; mm++)
	{
		random = rannew64(&idum);
		if (polyFlag) rad_aux = radMin + random*deltaRad;
		else if (deltaRad == 0.0) rad_aux = radMin;
		else if (random < 0.5) rad_aux = radMin;
		else rad_aux = radMin + deltaRad;

		massG = (4.0*PI/3.0)*rad_aux*rad_aux*rad_aux*rho_g;
		grainVec[mm].rad = rad_aux;
		grainVec[mm].mass = massG;
		grainVec[mm].inertia = (2.0/5.0)*massG*rad_aux*rad_aux;
		totMass += massG;
	}

	// Now compute Hopper dimensions. The origin is
	// located at a bottGap from the orifice.
	hopAngR = hopAng*PI/180.0;
	hopLength = 0.5*(siloWidth - hopWidth)/cos(hopAngR);
	siloInit = hopLength*sin(hopAngR) + bottGap;

	// Open the log and print properties
	fBit = fopen("bitacora", "w");
	fprintf(fBit, "Masa total (g) = %lf\n", totMass);
	fprintf(fBit, "K_gg = %lf; K_gw = %lf; K_gp = %lf\n",
		kappa_gg, kappa_gw, kappa_gp);
	fprintf(fBit, "G_gg = %lf; G_gw = %lf; G_gp = %lf\n",
		gamma_gg, gamma_gw, gamma_gp);

	// Pack parameters
	pars.siloWidth = siloWidth;
	pars.siloHeight = siloHeight;
	pars.siloThick = siloThick;
	pars.siloInit = siloInit;
	pars.hopWidth = hopWidth;
	pars.hopAngR = hopAngR;
	pars.hopLength = hopLength;
	pars.ngrains = ngrains;
	pars.gamma_gg = gamma_gg;
	pars.kappa_gg = kappa_gg;
	pars.mu_gg = mu_gg;
	pars.gamma_gw = gamma_gw;
	pars.kappa_gw = kappa_gw;
	pars.mu_gw = mu_gw;
	pars.gamma_gp = gamma_gp;
	pars.kappa_gp = kappa_gp;
	pars.mu_gp = mu_gp;
//	pars.xk_tn = xk_tn;
//	pars.xg_tn = xg_tn;
	pars.xk_tn = 2.0/7.0;
	pars.xg_tn = 1.0/3.0;
	pars.xmu_ds = xmu_ds;
	pars.dt = dt;
	pars.radMax = radMax;
	pars.bottGap = bottGap;
	pars.winWidth = winWidth;
	pars.tapOpen = 0;
	pars.tapOpenTime = tapOpenTime;
	pars.dischargeTarget = dischargeTarget;

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ INITIAL STATE +*+*+*+*+*+*+*+*+*+*+*+*+*/

	double3 *rrVec, *vvVec;
	double3 *wwVec;
	double *spawnTime;
	double3 *spawnPos;
	int *grainState;
	double theta, phi, jitter, xj, yj, zj;
	long lot_idx, lot_fill;

	// Get CPU-GPU memory in UNIFIED MEMORY
	cudaSafeCall(cudaMallocManaged(&rrVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&vvVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&wwVec, ngrains*sizeof(double3)));

	spawnTime = (double *) malloc(ngrains*sizeof(double));
	spawnPos = (double3 *) malloc(ngrains*sizeof(double3));
	grainState = (int *) malloc(ngrains*sizeof(int));
	if (!spawnTime || !spawnPos || !grainState)
	{
		printf("Error: memoria insuficiente para lotes\n");
		exit (3);
	}

	lot_fill = 0;
	for (lot_idx=0; lot_idx<lot_count; lot_idx++)
	{
		for (mm=0; mm<lots[lot_idx].count; mm++)
		{
			if (lot_fill >= ngrains)
			{
				printf("Error: demasiados granos en lotes\n");
				exit (3);
			}

			grainVec[lot_fill].lot_id = lots[lot_idx].lot_id;
			spawnTime[lot_fill] = lots[lot_idx].spawn_time;
			spawnPos[lot_fill] = lots[lot_idx].spawn_pos;
			grainState[lot_fill] = 0;
			lot_fill++;
		}
	}

	if (lot_fill != ngrains)
	{
		printf("Error: granos sin asignar a lotes\n");
		exit (3);
	}

	for (mm=0; mm<ngrains; mm++)
	{
		rrVec[mm].x = 0.0;
		rrVec[mm].y = 0.0;
		rrVec[mm].z = -1.0;

		vvVec[mm].x = 0.0;
		vvVec[mm].y = 0.0;
		vvVec[mm].z = 0.0;

		wwVec[mm].x = 0.0;
		wwVec[mm].y = 0.0;
		wwVec[mm].z = 0.0;
	}

	// Open files
	FILE *fSnap, *fQflow;

	if (snapFlag) fSnap = fopen("snapshots.xyz", "w");

	fQflow = fopen("qflow.dat", "w");
	fprintf(fQflow, "# Time\tQflow\n");

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ CELLS +*+*+*+*+*+*+*+*+*+*+*+*+*/

	long nCell_x, nCell_z, nCell2, nTags, nTot, nTouch;
	double vertHeight, cellSide_x, cellSide_z, arMin;

	/* Decide cuantas celdas poner. Usar celdas de diametros maximos (DIAMS).
	DIAMS tiene que ser al menos 1. Estima cuantos granos van en celda */

	// Number of horizontal and vertical cells
	nCell_x = (long)(siloWidth/(DIAMS*2.0*radMax));
	vertHeight = siloInit + siloHeight;
	nCell_z = (long)(vertHeight/(DIAMS*2.0*radMax));
	nCell2 = nCell_x*nCell_z;

	// Compute dimensions
	cellSide_x = siloWidth/(double) nCell_x;
	cellSide_z = vertHeight/(double) nCell_z;
	arMin = PI*radMin*radMin; // minimum area
	aux_0 = cellSide_x + 2.0*radMax;
	aux_1 = aux_0*(cellSide_z + 2.0*radMax);
	aux_0 = aux_1/arMin;
	nTags = (long) aux_0 + 1;
	nTot = nCell2*nTags;

	// Now compute the maximum number of contacts for a grain
	// by computing the number of diametrosMin (regular polygon)
	// that fit in a circle of radius radMax + radMin
	aux_0 = atan(radMin/(radMax + radMin));
	nTouch = (long)(PI/aux_0) + 3;

	// Write to the log
	fprintf(fBit, "CellSide = (%lf, %lf)\n", cellSide_x, cellSide_z);
	fprintf(fBit, "nCell = (%ld, %ld)\n", nCell_x, nCell_z);
	fprintf(fBit, "nTags = %ld\n", nTags);
	fprintf(fBit, "nTouch = %ld\n", nTouch);

	// Pack parameters
	pars.nCell_x = nCell_x;
	pars.nCell_z = nCell_z;
	pars.cellSide_x = cellSide_x;
	pars.cellSide_z = cellSide_z;
	pars.nCell2 = nCell2;
	pars.nTot = nTot;
	pars.nTags = nTags;
	pars.nTouch = nTouch;

	/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+ RUN +*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/

	touch *d_touchVec;
	int *d_idxReport;
	double3 *d_ffNewVec, *d_ffOldVec, *tmp_ff, *d_rrTolvVec,
		*d_ttNewVec, *d_ttOldVec, *tmp_tt, *rrcCntcVec, *ffcCntcVec;
	double *rrxProf, *vvzProf;
	long *phiHist, *d_nOcupVec, *d_cellVec, *nCntcVec, *tagCntcVec;
	long ths, blks, totTouch, thsTouch, blksTouch, thsBins, blksBins,
		nTrans,	nIter, ni, nGap, count = 0;
	double time, totTime, binSize, timeOld, clogTime;
	long qflow = 0;
	int flag, flagFin = 0;
	long lotDischarged[4] = {0, 0, 0, 0};
	long totalDischarged = 0;

	// Compute number of blocks and threads
	ths = (ngrains < THS_MAX) ? nextPow2(ngrains) : THS_MAX;
	blks = 1 + (ngrains - 1)/ths;

	totTouch = ngrains*nTouch;
	thsTouch = (totTouch < THS_MAX) ? nextPow2(totTouch) : THS_MAX;
	blksTouch = 1 + (totTouch - 1)/thsTouch;

	thsBins = (nBinsHop < THS_MAX) ? nextPow2(nBinsHop) : THS_MAX;
	blksBins = 1 + (nBinsHop - 1)/thsBins;

	time = -tTrans;
	nTrans = (long)(tTrans/dt);
	nIter = (long)(tRun/dt);
	nGap = (long) (1.0/(dt*gapFreq));
	totTime = nIter*dt;
	timeOld = 0.0;
	clogTime = 0.0;
	pars.tapOpen = (time >= pars.tapOpenTime) ? 1 : 0;

	// Compute the size of the orifice bins
	binSize = hopWidth/(double) nBinsHop;
	pars.binSize = binSize;
	pars.nBinsHop = nBinsHop;

	// Get CPU-GPU memory and GPU-only (device) memory
	cudaSafeCall(cudaMalloc(&d_nOcupVec, nCell2*sizeof(long)));
	cudaSafeCall(cudaMalloc(&d_cellVec, nTot*sizeof(long)));
	cudaSafeCall(cudaMalloc(&d_touchVec, totTouch*sizeof(touch)));
	cudaSafeCall(cudaMalloc(&d_ffNewVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMalloc(&d_ffOldVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMalloc(&d_ttNewVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMalloc(&d_ttOldVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMalloc(&d_rrTolvVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMalloc(&d_idxReport, ngrains*sizeof(int)));
	cudaSafeCall(cudaMallocManaged(&phiHist, nBinsHop*sizeof(long)));
	cudaSafeCall(cudaMallocManaged(&rrxProf, ngrains*sizeof(double)));
	cudaSafeCall(cudaMallocManaged(&vvzProf, ngrains*sizeof(double)));

	// For contacts
	cudaSafeCall(cudaMallocManaged(&nCntcVec, ngrains*sizeof(long)));
	cudaSafeCall(cudaMallocManaged(&tagCntcVec, totTouch*sizeof(long)));
	cudaSafeCall(cudaMallocManaged(&rrcCntcVec, totTouch*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&ffcCntcVec, totTouch*sizeof(double3)));


	// Clear vectors
	cudaMemset(phiHist, 0, nBinsHop*sizeof(long));
	cudaMemset(d_idxReport, 0, ngrains*sizeof(int));
	cudaMemset(d_nOcupVec, 0, nCell2*sizeof(long));
	cudaMemset(d_cellVec, -1, nTot*sizeof(long));

	cleanProf<<<blks, ths>>>(rrxProf, vvzProf, ngrains);
	cudaCheckError();

	clean<<<blks, ths>>>(d_ffOldVec, d_ttOldVec, grainVec, ngrains);
	cudaCheckError();

	cleanTouch<<<blksTouch, thsTouch>>>(d_touchVec, totTouch);
	cudaCheckError();

	// Locate grains in cells
	cellLocate<<<blks, ths>>>(rrVec, d_nOcupVec, d_cellVec, pars);
	cudaCheckError();

	getContacts<<<blks, ths>>>(rrVec, grainVec, d_cellVec,
		d_nOcupVec, nCntcVec, tagCntcVec, d_rrTolvVec, pars);
	cudaCheckError();

	getForces<<<blks, ths>>>(rrVec, vvVec, wwVec, d_ffOldVec,
		d_ttOldVec, grainVec, d_touchVec, nCntcVec,
		tagCntcVec, d_rrTolvVec, rrcCntcVec, ffcCntcVec, pars);
	cudaCheckError();

	for (ni=-nTrans; ni<=nIter; ni++)
	{
		flag = abs(ni)%nGap;

		pars.tapOpen = (time >= pars.tapOpenTime) ? 1 : 0;

		jitter = 0.25*radMax;
		for (mm=0; mm<ngrains; mm++)
		{
			if (grainState[mm] != 0) continue;
			if (time < spawnTime[mm]) continue;

			xj = (rannew64(&idum) - 0.5)*2.0*jitter;
			yj = (rannew64(&idum) - 0.5)*2.0*jitter;
			zj = (rannew64(&idum) - 0.5)*2.0*jitter;

			rrVec[mm].x = spawnPos[mm].x + xj;
			rrVec[mm].y = spawnPos[mm].y + yj;
			rrVec[mm].z = spawnPos[mm].z + zj;

			if (rrVec[mm].x < radMax) rrVec[mm].x = radMax;
			if (rrVec[mm].x > siloWidth - radMax)
				rrVec[mm].x = siloWidth - radMax;
			if (rrVec[mm].y < -0.5*siloThick + radMax)
				rrVec[mm].y = -0.5*siloThick + radMax;
			if (rrVec[mm].y > 0.5*siloThick - radMax)
				rrVec[mm].y = 0.5*siloThick - radMax;
			if (rrVec[mm].z < radMax) rrVec[mm].z = radMax;
			if (rrVec[mm].z > siloInit + siloHeight - radMax)
				rrVec[mm].z = siloInit + siloHeight - radMax;

			phi = 2.0*PI*rannew64(&idum);
			theta = PI*rannew64(&idum);
			vvVec[mm].x = v0*cos(phi)*sin(theta);
			vvVec[mm].y = v0*sin(phi)*sin(theta);
			vvVec[mm].z = v0*cos(theta);

			wwVec[mm].x = 0.0;
			wwVec[mm].y = 0.0;
			wwVec[mm].z = 0.0;

			grainState[mm] = 1;
		}

		verletInit<<<blks, ths>>>(rrVec, vvVec, wwVec, d_ffOldVec,
			d_ttOldVec, grainVec, pars, d_idxReport);
		cudaCheckError();

		cudaMemset(d_nOcupVec, 0, nCell2*sizeof(long));
		cudaMemset(d_cellVec, -1, nTot*sizeof(long));

		clean<<<blks, ths>>>(d_ffNewVec, d_ttNewVec, grainVec, ngrains);
		cudaCheckError();

		cleanTouch<<<blksTouch, thsTouch>>>(d_touchVec, totTouch);
		cudaCheckError();

		cellLocate<<<blks, ths>>>(rrVec, d_nOcupVec, d_cellVec, pars);
		cudaCheckError();

		getContacts<<<blks, ths>>>(rrVec, grainVec, d_cellVec,
			d_nOcupVec, nCntcVec, tagCntcVec, d_rrTolvVec, pars);
		cudaCheckError();

		getForces<<<blks, ths>>>(rrVec, vvVec, wwVec, d_ffNewVec,
			d_ttNewVec, grainVec, d_touchVec, nCntcVec,
			tagCntcVec, d_rrTolvVec, rrcCntcVec, ffcCntcVec, pars);
		cudaCheckError();

		verletFinish<<<blks, ths>>>(vvVec, wwVec, d_ffOldVec, d_ffNewVec,
			d_ttOldVec, d_ttNewVec, grainVec, pars);
		cudaCheckError();

		// Swap (new <--> old)
		tmp_ff = d_ffOldVec;
		d_ffOldVec = d_ffNewVec;
		d_ffNewVec = tmp_ff;

		tmp_tt = d_ttOldVec;
		d_ttOldVec = d_ttNewVec;
		d_ttNewVec = tmp_tt;

		// Advance time and update
		time += dt;

		pars.tapOpen = (time >= pars.tapOpenTime) ? 1 : 0;
		if (pars.tapOpen)
		{
			for (mm=0; mm<ngrains; mm++)
			{
				if (grainState[mm] != 1) continue;
				if (rrVec[mm].z >= 0.0) continue;

				grainState[mm] = 2;
				totalDischarged++;
				if (grainVec[mm].lot_id >= 1 && grainVec[mm].lot_id <= 3)
					lotDischarged[grainVec[mm].lot_id]++;
			}
		}

		if (pars.dischargeTarget > 0 &&
			totalDischarged >= pars.dischargeTarget)
		{
			printf("DESCARGA COMPLETA (%ld granos)\n", totalDischarged);
			break;
		}

		if (!flag)
		{
			cudaDeviceSynchronize();

			printf("Simulado %.4f de %.4f s\n", time, totTime);

			// Print in xyz format for visualization in Ovito
			if (snapFlag) xyzOvPrint(fSnap, grainVec, rrVec, radMin,
						radMax, pars);
		}

		if (ni < 0) continue;

		getVVprof<<<blks, ths>>>(rrVec, vvVec, d_idxReport, rrxProf,
			vvzProf, ngrains, siloWidth, hopWidth);
		cudaCheckError();

		qflow += thrust::reduce(thrust::device, d_idxReport,
				d_idxReport + ngrains, 0, thrust::plus<long>());

		if (qflow == 0) clogTime += dt;
		else if (qflow >= 80)
		{
			aux_0 = (double) qflow/(time - timeOld);
			fprintf(fQflow, "%lf\t%lf\n", time, aux_0);
			timeOld = time;
			qflow = 0;
			clogTime = 0.0;
		}

		if (flag) continue;

		// Add to phi histogram
		getPhi<<<blksBins, thsBins>>>(rrVec, grainVec, d_cellVec,
			d_nOcupVec, phiHist, pars);
		cudaCheckError();

		count++;

		cudaDeviceSynchronize();

		// Print data
		flagFin = dataPrint(grainVec, rrVec, vvVec, wwVec, nCntcVec,
		tagCntcVec, rrcCntcVec, ffcCntcVec, pars, count);

		// Stop if there are no grains in the hopper or if a jam
		// exists (no grains pass in 0.5 s)
		if (clogTime > 0.5)
		{
			printf("SE HA ATASCADO\n\n");
			break;
		}

		if (flagFin)
		{
			printf("NO HAY GRANOS EN LA VENTANA DE MEDICIÃ“N\n\n");
			break;
		}
	}

	cudaDeviceSynchronize();

	/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+ FINALIZE +*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/

	double xx_b, xcount, rrx_p, vvz_p;
	FILE *fPhi, *fVVprf, *fSnap_fin, *fLotsOut;

	fPhi = fopen("phiHist.dat", "w");
	fprintf(fPhi, "# rrxOrif\tphi\n");

	for (mm=0; mm<nBinsHop; mm++)
	{
		// Normalize distance by the orifice width
		xx_b = ((double) mm + 0.5)*binSize/hopWidth;

		// Average and write
		xcount = 1.0/(double) count;
		fprintf(fPhi, "%lf\t%lf\n", xx_b, phiHist[mm]*xcount);
	}

	fVVprf = fopen("vvProfile.dat", "w");
	fprintf(fVVprf, "# rrxOrif\tvvzGrano\n");

	for (mm=0; mm<ngrains; mm++)
	{
		rrx_p = rrxProf[mm];
		vvz_p = vvzProf[mm];
		if (rrx_p == 0.0) continue;

		fprintf(fVVprf, "%lf\t%lf\n", rrx_p, vvz_p);
	}

	// Print the last snapshot in xyz format
	fSnap_fin = fopen("last_snapshot.xyz", "w");
	xyzOvPrint(fSnap_fin, grainVec, rrVec, radMin, radMax, pars);

	fLotsOut = fopen("lot_discharge.dat", "w");
	fprintf(fLotsOut, "# lot_id\tcount\n");
	for (mm=1; mm<=3; mm++)
		fprintf(fLotsOut, "%ld\t%ld\n", mm, lotDischarged[mm]);
	fprintf(fLotsOut, "# total\t%ld\n", totalDischarged);

	// Close files
	fclose(fBit);
	if (snapFlag) fclose(fSnap);
	fclose(fPhi);
	fclose(fVVprf);
	fclose(fSnap_fin);
	fclose(fLotsOut);
	fclose(fQflow);

	// Free memory
	cudaFree(grainVec);
	cudaFree(rrVec);
	cudaFree(vvVec);
	cudaFree(wwVec);
	cudaFree(d_idxReport);
	cudaFree(rrxProf);
	cudaFree(vvzProf);
	cudaFree(phiHist);
	cudaFree(tagCntcVec);
	cudaFree(rrcCntcVec);
	cudaFree(ffcCntcVec);
	cudaFree(d_nOcupVec);
	cudaFree(d_cellVec);
	cudaFree(d_touchVec);
	cudaFree(d_ffNewVec);
	cudaFree(d_ffOldVec);
	cudaFree(d_ttNewVec);
	cudaFree(d_ttOldVec);
	cudaFree(nCntcVec);
	cudaFree(d_rrTolvVec);
	free(spawnTime);
	free(spawnPos);
	free(grainState);

	printf("TERMINADO\n");

	exit (0);
}





