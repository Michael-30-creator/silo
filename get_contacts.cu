#include "estructuras.h"

__global__ void getContacts(double3 *rrVec, grain_prop *grainVec,
			long *cellVec, long *nOcupVec, long *nCntcVec,
			long *tagCntcVec, double3 *rrTolvVec,
			parameters pars)
{
	double3 rra, rrb, drr, hop, rrah, rrbh;
	long ngrains, nCell_x, nCell_z, nTags, nTouch,
		ii, jj, iib, jjb, idel, jdel, idx,
		tag, tag_init, tag_end, bb;
	long indCntc_init, nCntc, indCntc;
	double cellSide_x, cellSide_z, siloWidth, siloInit, siloHeight,
		h_siloThick, rad_a, rad_b, sum_radii, sum_radii2, dist,
		dist2, hopWidth, hopLength, hopAngR, dot, param;

	ngrains = pars.ngrains;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		rra = rrVec[ind];
		rad_a = grainVec[ind].rad;

		nTouch = pars.nTouch;
		indCntc_init = ind*nTouch;
		nCntc = 0;

		if (rra.z < 0.0)
		{
			nCntcVec[ind] = nCntc;
			return;
		}

		nCell_x = pars.nCell_x;
		cellSide_x = pars.cellSide_x;
		ii = (long)(rra.x/cellSide_x);
		if (ii == nCell_x) ii--;

		nCell_z = pars.nCell_z;
		cellSide_z = pars.cellSide_z;
		jj = (long)(rra.z/cellSide_z);
		if (jj == nCell_z) jj--;

		/*+*+*+*+*+*+*+* GRANO-PARED_IZQUIERDA +*+*+*+*+*+*+*/

		if (ii == 0)
		{
			dist = rra.x;

			if (dist < rad_a)
			{
				//Contacto
				bb = ngrains + 1;
				indCntc = indCntc_init + nCntc;
				tagCntcVec[indCntc] = bb;
				nCntc++;
			}
		}

		/*+*+*+*+*+*+*+* GRANO-PARED_DERECHA +*+*+*+*+*+*+*/

		siloWidth = pars.siloWidth;

		if (ii == nCell_x - 1)
		{
			dist = siloWidth - rra.x;

			if (dist < rad_a)
			{
				//Contacto
				bb = ngrains + 2;
				indCntc = indCntc_init + nCntc;
				tagCntcVec[indCntc] = bb;
				nCntc++;
			}
		}

		/*+*+*+*+*+*+*+* GRANO-TOLVA +*+*+*+*+*+*+*/

		siloInit = pars.siloInit;
		hopLength = pars.hopLength;
		hopAngR = pars.hopAngR;
		hopWidth = pars.hopWidth;

		if (rra.z < siloInit + rad_a)
		{
			// Coordenada z del punto final del hopper
			hop.z = -hopLength*sin(hopAngR);

			if (rra.z > siloInit + hop.z - rad_a)
			{
				/*+*+*+*+*+*+*+ IZQUIERDA *+*+*+*+*+*+*/

				// Coordenada x del punto final del hopper
				hop.x = 0.5*(siloWidth - hopWidth);

				if (rra.x < hop.x + rad_a)
				{
					// Pone el grano con referencia
					// al punto inicial del hopper
					rrah.x = rra.x;
					rrah.z = rra.z - siloInit;

					// Pruducto punto
					dot = rrah.x*hop.x + rrah.z*hop.z;
					param = dot/(hopLength*hopLength);

					if (param < 0.0) param = 0.0;
					if (param > 1.0) param = 1.0;

					// Punto mas cercano al grano
					rrbh.x = param*hop.x;
					rrbh.z = param*hop.z;

					// Pone en el origen de coordenadas
					rrb.x = rrbh.x;
					rrb.y = rra.y;
					rrb.z = rrbh.z + siloInit;

					// Ahora checa distancias
					drr.x = rrb.x - rra.x;
					drr.y = rrb.y - rra.y;
					drr.z = rrb.z - rra.z;
					dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;

					if (dist2 < rad_a*rad_a)
					{
						//Contacto
						bb = ngrains + 3;
						rrTolvVec[ind] = rrb;
						indCntc = indCntc_init + nCntc;
						tagCntcVec[indCntc] = bb;
						nCntc++;
					}
				}

				/*+*+*+*+*+*+*+* DERECHA +*+*+*+*+*+*+*/

				// Coordenada x del punto final del hopper
				hop.x = -hop.x;

				if (rra.x > siloWidth + hop.x - rad_a)
				{
					// Pone el grano con referencia
					// al punto inicial del hopper
					rrah.x = rra.x - siloWidth;
					rrah.z = rra.z - siloInit;

					// Pruducto punto
					dot = rrah.x*hop.x + rrah.z*hop.z;
					param = dot/(hopLength*hopLength);

					if (param < 0.0) param = 0.0;
					if (param > 1.0) param = 1.0;

					// Punto mas cercano al granos
					rrbh.x = param*hop.x;
					rrbh.z = param*hop.z;

					// Pone en el origen de coordenadas
					rrb.x = rrbh.x + siloWidth;
					rrb.y = rra.y;
					rrb.z = rrbh.z + siloInit;

					// Ahora checa distancias
					drr.x = rrb.x - rra.x;
					drr.y = rrb.y - rra.y;
					drr.z = rrb.z - rra.z;
					dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;

					if (dist2 < rad_a*rad_a)
					{
						//Contacto
						bb = ngrains + 4;
						rrTolvVec[ind] = rrb;
						indCntc = indCntc_init + nCntc;
						tagCntcVec[indCntc] = bb;
						nCntc++;
					}
				}
			}
		}

		/*+*+*+*+*+*+*+* GRANO-PLANO_POSTERIOR +*+*+*+*+*+*+*/

		h_siloThick = 0.5*pars.siloThick;
		if (rra.y >= 0.0)
		{
			dist = h_siloThick - rra.y;
			if (dist < rad_a)
			{
				//Contacto
				bb = ngrains + 5;
				indCntc = indCntc_init + nCntc;
				tagCntcVec[indCntc] = bb;
				nCntc++;
			}
		}

		/*+*+*+*+*+*+*+* GRANO-PLANO_FRONTAL +*+*+*+*+*+*+*/

		if (rra.y <= 0.0)
		{
			dist = h_siloThick + rra.y;
			if (dist < rad_a)
			{
				//Contacto
				bb = ngrains + 6;
				indCntc = indCntc_init + nCntc;
				tagCntcVec[indCntc] = bb;
				nCntc++;
			}
		}

		/*+*+*+*+*+*+*+* GRANO-TECHO +*+*+*+*+*+*+*/

		siloHeight = pars.siloHeight;
		if (jj == nCell_z - 1)
		{
			dist = siloInit + siloHeight - rra.z;
			if (dist < rad_a)
			{
					//Contacto
					bb = ngrains + 7;
					indCntc = indCntc_init + nCntc;
					tagCntcVec[indCntc] = bb;
					nCntc++;
			}
		}

		/*+*+*+*+*+*+*+* GRANO-PISO +*+*+*+*+*+*+*/

//		if (jj == 0)
//		{
//			dist = rra.z;
//			if (dist < rad_a)
//			{
//					//Contacto
//					bb = ngrains + 8;
//					indCntc = indCntc_init + nCntc;
//					tagCntcVec[indCntc] = bb;
//					nCntc++;
//			}
//		}

		/*+*+*+*+*+*+*+*+*+* GRANO-GRANO +*+*+*+*+*+*+*+*+*/

		nTags = pars.nTags;

		// Checa las 9 celdas del vecindario, si existen
		for (idel=-1; idel<=1; idel++)
		for (jdel=-1; jdel<=1; jdel++) 
		{ 
			iib = ii + idel;
			if (iib < 0 || iib >= nCell_x) continue;
			jjb = jj + jdel;
			if (jjb < 0 || jjb >= nCell_z) continue;

			// Busca los granos vecinos
			idx = iib + jjb*nCell_x;
			tag_init = idx*nTags;
			tag_end = tag_init + nOcupVec[idx];

			for(tag=tag_init; tag<tag_end; tag++)
			{
				bb = cellVec[tag];
				if (bb == ind) continue;

				rrb = rrVec[bb];
				rad_b = grainVec[bb].rad;

				sum_radii = rad_a + rad_b;
				sum_radii2 = sum_radii*sum_radii;

				// Checa distancias
				drr.x = rrb.x - rra.x;
				drr.y = rrb.y - rra.y;
				drr.z = rrb.z - rra.z;
				dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;

				if (dist2 >= sum_radii2) continue;

				//Contacto
				indCntc = indCntc_init + nCntc;
				tagCntcVec[indCntc] = bb;
				nCntc++;
			}
		}

		nCntcVec[ind] = nCntc;
	}

	return;
}
