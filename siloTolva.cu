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

// Generador de números aleatorios uniformes entre 0.0 a 1.0
double rannew64(long*);

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

	// Imprime en formato .xyz para Ovito
	fprintf(fSnap, "%ld\n", npart + 6);
	fprintf(fSnap, "Lattice=");
	fprintf(fSnap, "\"%.2f 0.00 0.00 ", siloWidth);
	fprintf(fSnap, "0.00 %.2f 0.00 ", siloThick);
	fprintf(fSnap, "0.00 0.00 %.2f\"\n", boxDim);

	// Imprime vertices
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

	// Imprime granos
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

	// Imprime granos
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

// Encuentra la siguiente potencia de dos
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
	/*+*+*+*+*+*+*+*+*+*+*+*+*+ PARÁMETROS +*+*+*+*+*+*+*+*+*+*+*+*+*/

	double siloWidth, siloHeight, siloThick, hopWidth, hopAng,
		bottGap, diamMin, diamMax, rho_g, rho_w, rho_p,
		tColl, eps_gg, mu_gg, eps_gw, mu_gw, eps_gp, mu_gp,
		xk_tn, xg_tn, xmu_ds, dt, gapFreq, tRun, tTrans, v0,
		winWidth;
	long ngrains, idum, nBinsHop;
	int polyFlag, snapFlag, err_flag = 0;
	char renglon[200];
	struct stat dirStat;

	// Ancho del silo; Altura del silo
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf %lf %lf", &siloWidth, &siloHeight, &siloThick);

	// Ancho del orificio; Ángulo de la tolva; Gap del orificio
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf %lf", &hopWidth, &bottGap);

	// Ángulo de la tolva
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
        else sscanf(renglon, "%lf", &hopAng);

	// Número de granos
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &ngrains);

	// ¿Polidispersidad?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &polyFlag);

	// Radio de grano
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &diamMin);

	// Diferencia de radios
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &diamMax);

	// Densidad
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf %lf", &rho_g, &rho_w, &rho_p);

	// Tiempo de colisión
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tColl);

	// Epsilón y mu grano-grano
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gg, &mu_gg);

	// Epsilón y mu grano-pared
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gw, &mu_gw);

	// Epsilón y mu grano-plano
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf", &eps_gp, &mu_gp);

	// Relación Kappa_t/Kappa_n; Gamma_t/Gamma_n; Mu_d/Mu_s
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf %lf %lf", &xk_tn, &xg_tn, &xmu_ds);

	// Paso de tiempo
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &dt);

	// Gap de impresión en dt's
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &gapFreq);

	// Tiempo de simulación
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tRun);

	// Tiempo transiente
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tTrans);

	// Velocidad inicial
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &v0);

	// Ancho de la ventana de medición
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &winWidth);

	// ¿Imprime snapshots?
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &snapFlag);

	// Semilla para números aleatorios
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &idum);

	// Número de bins en el orificio
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%ld", &nBinsHop);

	if (err_flag)
	{
		printf("Error en el archivo (.data) de parámetros.\n");
		exit (1);
	}

	if (siloThick < diamMax)
	{
		printf("Error: El espesor del silo no puede ser menor"
			" que el diametro máximo de grano.\n");
		exit (2);
	}

	if (stat("DataFrames", &dirStat) == -1) mkdir("DataFrames", 0700);

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ PROPIEDADES +*+*+*+*+*+*+*+*+*+*+*+*+*/

	grain_prop *grainVec;
	parameters pars;
	long mm;
	double radAve, massAve_g, massAve_w, massAve_p, mEff,
		aux_0, aux_1, gamma_gg, kappa_gg, gamma_gw,
		kappa_gw, gamma_gp, kappa_gp, deltaRad, random,
		rad_aux, massG,	totMass;
	double hopAngR, hopLength, radMin, radMax, siloInit;
	FILE *fBit;

	// Calcula masas promedio
	radMin = diamMin/2.0;
	radMax = diamMax/2.0;
	deltaRad = (diamMax - diamMin)/2.0;
	radAve = radMin + 0.5*deltaRad;
	massAve_g = (4.0*PI/3.0)*radAve*radAve*radAve*rho_g;
	massAve_w = (4.0*PI/3.0)*radAve*radAve*radAve*rho_w;
	massAve_p = (4.0*PI/3.0)*radAve*radAve*radAve*rho_p;

	// Calcula kappa y gamma Grano-Grano
	mEff = massAve_g/2.0;
	gamma_gg = -2.0*mEff*log(eps_gg)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gg)/tColl;
	kappa_gg = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Calcula kappa y gamma Grano-Pared
	mEff = massAve_g*massAve_w/(massAve_g + massAve_w);
	gamma_gw = -2.0*mEff*log(eps_gw)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gw)/tColl;
	kappa_gw = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Calcula kappa y gamma Grano-Plano
	mEff = massAve_g*massAve_p/(massAve_g + massAve_p);
	gamma_gp = -2.0*mEff*log(eps_gp)/tColl;
	aux_0 = PI/tColl;
	aux_1 = log(eps_gp)/tColl;
	kappa_gp = mEff*(aux_0*aux_0 + aux_1*aux_1);

	// Consigue memoria CPU-GPU en la MEMORIA UNIFICADA
	cudaSafeCall(cudaMallocManaged(&grainVec, ngrains*sizeof(grain_prop)));

	// Calcula radios, masas, inercias y guarda
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

	// Ahora calcula dimensiones del Hopper. El origen se
	// encontrara a un bottGap del orificio.
	hopAngR = hopAng*PI/180.0;
	hopLength = 0.5*(siloWidth - hopWidth)/cos(hopAngR);
	siloInit = hopLength*sin(hopAngR) + bottGap;

	// Abre la bitacora e imprime propiedades
	fBit = fopen("bitacora", "w");
	fprintf(fBit, "Masa total (g) = %lf\n", totMass);
	fprintf(fBit, "K_gg = %lf; K_gw = %lf; K_gp = %lf\n",
		kappa_gg, kappa_gw, kappa_gp);
	fprintf(fBit, "G_gg = %lf; G_gw = %lf; G_gp = %lf\n",
		gamma_gg, gamma_gw, gamma_gp);

	// Empaca parámetros
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

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ ESTADO INICIAL +*+*+*+*+*+*+*+*+*+*+*+*+*/

	double3 *rrVec, *vvVec;
	double3 *wwVec;
	double xx, zz, shift, xxInit, zzInit, xxFin, zzFin, theta, phi;
	long rowCounter;

	// Consigue memoria CPU-GPU en la MEMORIA UNIFICADA
	cudaSafeCall(cudaMallocManaged(&rrVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&vvVec, ngrains*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&wwVec, ngrains*sizeof(double3)));

	// Los granos se colocan inicialmente en el silo.
	zz = siloInit;
	shift = 1.05*radMax;
	xxInit = shift;
	xxFin = siloWidth - shift;
	zzInit = 2.8*shift;
	zzFin = siloInit + siloHeight - shift;
	xx = xxInit;
	zz += zzInit;

	// Hace un llenado inicial en formación hexagonal
	mm = 0;
	rowCounter = 0;

	while (1)
	{
		rrVec[mm].x = xx;
		rrVec[mm].y = 0.0;
		rrVec[mm].z = zz;

		phi = 2.0*PI*rannew64(&idum);
		theta = PI*rannew64(&idum);
		vvVec[mm].x = v0*cos(phi)*sin(theta);
		vvVec[mm].y = v0*sin(phi)*sin(theta);
		vvVec[mm].z = v0*cos(theta);

		wwVec[mm].x = 0.0;
		wwVec[mm].y = 0.0;
		wwVec[mm].z = 0.0;

		mm++;
		if (mm == ngrains) break;

		// Nuevo punto
		xx += 2.0*shift;

		if (xx < xxFin) continue;

		zz += 1.74*shift;
		rowCounter++;
		if (rowCounter%2 == 0) xx = xxInit;
		else xx = xxInit + shift;

		if (zz > zzFin)
		{
			printf("Error: el número máximo de granos "
				"para este sistema es %ld\n", mm);
			exit (3);
		}
	}

	// Abre archivos
	FILE *fSnap, *fQflow;

	if (snapFlag) fSnap = fopen("snapshots.xyz", "w");

	fQflow = fopen("qflow.dat", "w");
	fprintf(fQflow, "# Time\tQflow\n");

	/*+*+*+*+*+*+*+*+*+*+*+*+*+ CELDAS +*+*+*+*+*+*+*+*+*+*+*+*+*/

	long nCell_x, nCell_z, nCell2, nTags, nTot, nTouch;
	double vertHeight, cellSide_x, cellSide_z, arMin;

	/* Decide cuantas celdas poner. Usar celdas de diametros maximos (DIAMS).
	DIAMS tiene que ser al menos 1. Estima cuantos granos van en celda */

	// Número de celdas horizontales y verticales
	nCell_x = (long)(siloWidth/(DIAMS*2.0*radMax));
	vertHeight = siloInit + siloHeight;
	nCell_z = (long)(vertHeight/(DIAMS*2.0*radMax));
	nCell2 = nCell_x*nCell_z;

	// Calcula dimensiones
	cellSide_x = siloWidth/(double) nCell_x;
	cellSide_z = vertHeight/(double) nCell_z;
	arMin = PI*radMin*radMin; //área mínima
	aux_0 = cellSide_x + 2.0*radMax;
	aux_1 = aux_0*(cellSide_z + 2.0*radMax);
	aux_0 = aux_1/arMin;
	nTags = (long) aux_0 + 1;
	nTot = nCell2*nTags;

	// Ahora calcua el número máximo de contactos en un grano
	// calculando el número de diametrosMin (poligono regular)
	// que caben en una circunferencia de radio radMax + radMin
	aux_0 = atan(radMin/(radMax + radMin));
	nTouch = (long)(PI/aux_0) + 3;

	// Escribe en la bitacora
	fprintf(fBit, "CellSide = (%lf, %lf)\n", cellSide_x, cellSide_z);
	fprintf(fBit, "nCell = (%ld, %ld)\n", nCell_x, nCell_z);
	fprintf(fBit, "nTags = %ld\n", nTags);
	fprintf(fBit, "nTouch = %ld\n", nTouch);

	// Empaca parámetros
	pars.nCell_x = nCell_x;
	pars.nCell_z = nCell_z;
	pars.cellSide_x = cellSide_x;
	pars.cellSide_z = cellSide_z;
	pars.nCell2 = nCell2;
	pars.nTot = nTot;
	pars.nTags = nTags;
	pars.nTouch = nTouch;

	/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+ CORRE +*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/

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

	// Calcula número de bloques e hilos
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

	// Calcula el tamaño de los bins del orificio
	binSize = hopWidth/(double) nBinsHop;
	pars.binSize = binSize;
	pars.nBinsHop = nBinsHop;

	// Consigue memoria CPU-GPU y solo GPU (device)
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

	// Para contactos
	cudaSafeCall(cudaMallocManaged(&nCntcVec, ngrains*sizeof(long)));
	cudaSafeCall(cudaMallocManaged(&tagCntcVec, totTouch*sizeof(long)));
	cudaSafeCall(cudaMallocManaged(&rrcCntcVec, totTouch*sizeof(double3)));
	cudaSafeCall(cudaMallocManaged(&ffcCntcVec, totTouch*sizeof(double3)));


	// Limpia vectores
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

	// Localiza los granos en las celdas
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

		// Intercambia (new <--> old)
		tmp_ff = d_ffOldVec;
		d_ffOldVec = d_ffNewVec;
		d_ffNewVec = tmp_ff;

		tmp_tt = d_ttOldVec;
		d_ttOldVec = d_ttNewVec;
		d_ttNewVec = tmp_tt;

		// Aumenta el tiempo y actualiza
		time += dt;

		if (!flag)
		{
			cudaDeviceSynchronize();

			printf("Simulado %.4f de %.4f s\n", time, totTime);

			// Imprime en formato xyz para visualizar en Ovito
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

		// Suma al histograma de phi
		getPhi<<<blksBins, thsBins>>>(rrVec, grainVec, d_cellVec,
			d_nOcupVec, phiHist, pars);
		cudaCheckError();

		count++;

		cudaDeviceSynchronize();

		// Imprime datos
		flagFin = dataPrint(grainVec, rrVec, vvVec, wwVec, nCntcVec,
		tagCntcVec, rrcCntcVec, ffcCntcVec, pars, count);

		// Finaliza si ya no hay granos en la tolva o si existe algun
		// atasco (no pasan granos en 0.5 s)
		if (clogTime > 0.5)
		{
			printf("SE HA ATASCADO\n\n");
			break;
		}

		if (flagFin)
		{
			printf("NO HAY GRANOS EN LA VENTANA DE MEDICIÓN\n\n");
			break;
		}
	}

	cudaDeviceSynchronize();

	/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+ FINALIZA +*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/

	double xx_b, xcount, rrx_p, vvz_p;
	FILE *fPhi, *fVVprf, *fSnap_fin;

	fPhi = fopen("phiHist.dat", "w");
	fprintf(fPhi, "# rrxOrif\tphi\n");

	for (mm=0; mm<nBinsHop; mm++)
	{
		// Normaliza la distancia con el ancho del orificio
		xx_b = ((double) mm + 0.5)*binSize/hopWidth;

		// Promedia y escribe
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

	// Imprime el ultimo snapchot en formato xyz
	fSnap_fin = fopen("last_snapshot.xyz", "w");
	xyzOvPrint(fSnap_fin, grainVec, rrVec, radMin, radMax, pars);

	// Cierra archivos
	fclose(fBit);
	if (snapFlag) fclose(fSnap);
	fclose(fPhi);
	fclose(fVVprf);
	fclose(fSnap_fin);
	fclose(fQflow);

	// Libera memoria
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

	printf("TERMINADO\n");

	exit (0);
}
