#include "estructuras.h"

__global__ void clean(double3 *ffVec, double3 *ttVec, grain_prop *grainVec,
			long ngrains)
{
	double mass;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		mass = grainVec[ind].mass;

		ffVec[ind].x = 0.0;
		ffVec[ind].y = 0.0;
		ffVec[ind].z = -GRAV*mass;

		ttVec[ind].x = 0.0;
		ttVec[ind].y = 0.0;
		ttVec[ind].z = 0.0;
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void cleanTouch(touch *touchVec, long totTouch)
{
	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < totTouch)
	{
		if (!touchVec[ind].flag)
		{
			touchVec[ind].tag = -1;
			touchVec[ind].dynF = 0;
		}

		touchVec[ind].flag = 0;
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void cleanProf(double *rrxProf, double *vvzProf, long ngrains)
{
	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		rrxProf[ind] = 0.0;
		vvzProf[ind] = 0.0;
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void cellLocate(double3 *rrVec, long *nOcupVec,
				long *cellVec, parameters pars)
{
	double3 rr;
	double cellSide_x, cellSide_z;
	long ii, jj, cellIndex, ngrains, nTags,
		nCell_x, nCell_z, shift;

	ngrains = pars.ngrains;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		rr = rrVec[ind];

		if (rr.z < 0.0) return;

		cellSide_x = pars.cellSide_x;
		nCell_x = pars.nCell_x;
		ii = (long)(rr.x/cellSide_x);
		if (ii == nCell_x) ii--;

		cellSide_z = pars.cellSide_z;
		nCell_z = pars.nCell_z;
		jj = (long)(rr.z/cellSide_z);
		if (jj == nCell_z) jj--;

		cellIndex = ii + jj*nCell_x;

		// Agrega uno a n_ocup.
		shift = atomicAdd((int *)&nOcupVec[cellIndex], 1);

		// Guarda índice de granos
		nTags = pars.nTags;
		cellVec[cellIndex*nTags + shift] = ind;
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void getPhi(double3 *rrVec, grain_prop *grainVec,long *cellVec,
		long *nOcupVec, long *phiHist, parameters pars)
{
	double3 rra;
	long nBinsHop, nTags, nCell_x, iib_i, iib_f, jjb, iia, jja,
		idx, tag_init, tag_end, tag, aa;
	double siloWidth, hopWidth, binSize, bottGap, cellSide_x, cellSide_z,
		rrbz, rrbx_i, rrbx_f, xind, rad_a, rad2, h_dist;

	nBinsHop = pars.nBinsHop;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < nBinsHop)
	{
		siloWidth = pars.siloWidth;
		hopWidth = pars.hopWidth;
		binSize = pars.binSize;
		bottGap = pars.bottGap;
		nCell_x = pars.nCell_x;
		nTags = pars.nTags;

		// Ubica la posición del bin en el orificio
		xind = (double) ind;
		rrbz = bottGap;
		rrbx_i = 0.5*(siloWidth - hopWidth) + xind*binSize;
		rrbx_f = rrbx_i + binSize;

		// Encuentra entre que celdas se encuentra el bin
		cellSide_x = pars.cellSide_x;
		iib_i = (long)(rrbx_i/cellSide_x);
		iib_f = (long)(rrbx_f/cellSide_x);

		cellSide_z = pars.cellSide_z;
		jjb = (long)(rrbz/cellSide_z);

		// Checa las celdas del vecindario
		for (iia = iib_i - 1; iia <= iib_f + 1; iia++)
		for (jja = jjb - 1; jja <= jjb + 1; jja++)
		{ 
			// Busca los granos vecinos
			idx = iia + jja*nCell_x;
			tag_init = idx*nTags;
			tag_end = tag_init + nOcupVec[idx];

			for(tag=tag_init; tag<tag_end; tag++)
			{
				aa = cellVec[tag];
				rra = rrVec[aa];
				rad_a = grainVec[aa].rad;

				// Evalua si esta fuera del rango
				// vertical del bin
				if (rra.z <= rrbz - rad_a) continue;
				if (rra.z >= rrbz + rad_a) continue;

				// Evalua si esta dentro del rango
				// horizontal del bin
				if (rra.x > rrbx_i && rra.x < rrbx_f)
				{
					// Si esta dentro lo reporta
					phiHist[ind] += 1;

					return;
				}

				// Evalua si esta fuera del rango
				// horizontal del bin
				if (rra.x <= rrbx_i - rad_a) continue;
				if (rra.x >= rrbx_f + rad_a) continue;

				// Calcula la semidistancia que esta
				// en la linea del orifico
				rad2 = rad_a*rad_a;
				h_dist = sqrt(rad2 - rra.z*rra.z);

				// Ahora evalua si no toca el bin
				if (rra.x + h_dist < rrbx_i) continue;
				if (rra.x - h_dist >= rrbx_f) continue;

				// Si llega hasta aqui es que el grano
				// toca al bin, asi que lo reporta
				phiHist[ind] += 1;

				return;
			}
		}
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void getVVprof(double3 *rrVec, double3 *vvVec,
		int *idxReport, double *rrxProf, double *vvzProf,
		long ngrains, double siloWidth, double hopWidth)
{
	double rrx;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind<ngrains)
	{
		if (!idxReport[ind]) return;

		rrx = rrVec[ind].x - 0.5*(siloWidth - hopWidth);
		rrx /= hopWidth;
		rrxProf[ind] = rrx;
		vvzProf[ind] = vvVec[ind].z;
	}

	return;
}
