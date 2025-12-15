#include "estructuras.h"

__global__ void verletInit(double3 *rrVec, double3 *vvVec, double3 *wwVec,
		double3 *ffVec, double3 *ttVec, grain_prop *grainVec,
		parameters pars, int *idxReport)
{
	double3 rr, vv, ww, ff, tt;
	double dt, hdt, mass, inertia, rrzOld, hopH;
	long ngrains;

	ngrains = pars.ngrains;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;
  
	if (ind < ngrains)
	{
		// Fetch
		rr = rrVec[ind];
		rrzOld = rr.z;

		if (rr.z < 0.0) return;

		dt = pars.dt;
		hdt = 0.5*dt;

		vv = vvVec[ind];
		ww = wwVec[ind];

		ff = ffVec[ind];
		tt = ttVec[ind];

		mass = grainVec[ind].mass;
		inertia = grainVec[ind].inertia;

		// Actualiza
		rr.x += dt*(vv.x + hdt*ff.x/mass);
		rr.y += dt*(vv.y + hdt*ff.y/mass);
		rr.z += dt*(vv.z + hdt*ff.z/mass);

		hopH = pars.bottGap;
		if (rrzOld > hopH && rr.z < hopH) idxReport[ind] = 1;
		else idxReport[ind] = 0;

		// Prediccion de velocidades
		vv.x += dt*ff.x/mass;
		vv.y += dt*ff.y/mass;
		vv.z += dt*ff.z/mass;

		ww.x += dt*tt.x/inertia;
		ww.y += dt*tt.y/inertia;
		ww.z += dt*tt.z/inertia;

		// save
		rrVec[ind] = rr;
		vvVec[ind] = vv;
		wwVec[ind] = ww;
	}

	return;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void verletFinish(double3 *vvVec, double3 *wwVec, double3 *ffOldVec,
		double3 *ffNewVec, double3 *ttOldVec, double3 *ttNewVec,
		grain_prop *grainVec, parameters pars)
{
	double3 vv, ww, ffn, ffo, ttn, tto;
	double hdt, mass, inertia;
	long ngrains;

	ngrains = pars.ngrains;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		// fetch
		vv = vvVec[ind];
		ww = wwVec[ind];

		ffn = ffNewVec[ind];
		ttn = ttNewVec[ind];

		ffo = ffOldVec[ind];
		tto = ttOldVec[ind];

		mass = grainVec[ind].mass;
		inertia = grainVec[ind].inertia;

		hdt = 0.5*pars.dt;

		// Corrige velocidades (incluyendo la angular)
		vv.x += hdt*(ffn.x - ffo.x)/mass;
		vv.y += hdt*(ffn.y - ffo.y)/mass;	  
		vv.z += hdt*(ffn.z - ffo.z)/mass;

		ww.x += hdt*(ttn.x - tto.x)/inertia;
		ww.y += hdt*(ttn.y - tto.y)/inertia;
		ww.z += hdt*(ttn.z - tto.z)/inertia;	

		// save
		vvVec[ind] = vv;
		wwVec[ind] = ww;
	}

	return;
}
