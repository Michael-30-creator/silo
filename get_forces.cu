#include "estructuras.h"

__device__ int force_and_torque(double3 rra, double3 vva, double3 wwa, double3 rrb,
		double3 vvb, double3 wwb, double3 rrc, double3 nn, double comp,
		double kappa, double gamma, double mu, double xk, double xg,
		double xmu, long ind, long nTouch, long bb, double dt,
		touch *touchVec, double3 *pff, double3 *ptt)
{
	double3 vvac, vvbc, dvc, dvct, ff, tt, ss;
	double3 zeta, ff_tan;
	double3 zetaOld, nnOld, dOm;
	double dvcn, comp_dot, normal, tan_max,
		kappa_t, gamma_t, tangential, aux;
	long cc, idx;
	int found_flag;

	//Construye velocidades en el punto de contacto (v = w x r)
	vvac.x = vva.x + wwa.y*(rrc.z - rra.z) - wwa.z*(rrc.y - rra.y);
	vvac.y = vva.y + wwa.z*(rrc.x - rra.x) - wwa.x*(rrc.z - rra.z);
	vvac.z = vva.z + wwa.x*(rrc.y - rra.y) - wwa.y*(rrc.x - rra.x);

	vvbc.x = vvb.x + wwb.y*(rrc.z - rrb.z) - wwb.z*(rrc.y - rrb.y);
	vvbc.y = vvb.y + wwb.z*(rrc.x - rrb.x) - wwb.x*(rrc.z - rrb.z);
	vvbc.z = vvb.z + wwb.x*(rrc.y - rrb.y) - wwb.y*(rrc.x - rrb.x);

	//Delta velocidades
	dvc.x = vvbc.x - vvac.x;
	dvc.y = vvbc.y - vvac.y;
	dvc.z = vvbc.z - vvac.z;

	//Magnitud dvcn = dvc * n
	dvcn = dvc.x*nn.x + dvc.y*nn.y + dvc.z*nn.z;

	//Derivada de la distancia
	comp_dot = -dvcn;

	//Obtiene la Normal
	normal = kappa*comp + gamma*comp_dot;
	if (normal <= 0.0) return 0; //No hace nada si es atractiva

	ff.x = -normal*nn.x;
	ff.y = -normal*nn.y;
	ff.z = -normal*nn.z;

	if (mu == 0.0)
	{
		pff->x = ff.x;
		pff->y = ff.y;
		pff->z = ff.z;

		ptt->x = 0.0;
		ptt->y = 0.0;
		ptt->z = 0.0;

		return 1; // Hasta aqui llega si no hay fricción
	}

	// Ahora la fricción
	dvct.x = dvc.x - dvcn*nn.x;
	dvct.y = dvc.y - dvcn*nn.y;
	dvct.z = dvc.z - dvcn*nn.z;

	tan_max = mu*normal;
	kappa_t = xk*kappa;
	gamma_t = xg*gamma;

	// Evalua si ya existía un contacto y calcula f_t
	found_flag = 0;
	for (cc=0; cc<nTouch; cc++)
	{
		idx = cc + ind*nTouch;
		if (touchVec[idx].tag != bb) continue;

		found_flag = 1;
		touchVec[idx].flag = 1;
		zetaOld = touchVec[idx].zeta;
		nnOld = touchVec[idx].nn;

		// Obtiene el vector angulo de rotación (nn x nnOld)
		dOm.x = nn.y*nnOld.z - nn.z*nnOld.y;
		dOm.y = nn.z*nnOld.x - nn.x*nnOld.z;
		dOm.z = nn.x*nnOld.y - nn.y*nnOld.x;

		// Ahora rota antes de actualizar (zeta = zetaOld +  zetaOld x dOm)
		zeta.x = zetaOld.x + zetaOld.y*dOm.z - zetaOld.z*dOm.y;
		zeta.y = zetaOld.y + zetaOld.z*dOm.x - zetaOld.x*dOm.z;
		zeta.z = zetaOld.z + zetaOld.x*dOm.y - zetaOld.y*dOm.x;

		// Actualiza
		zeta.x += dvct.x*dt;
		zeta.y += dvct.y*dt;
		zeta.z += dvct.z*dt;

		ff_tan.x = kappa_t*zeta.x + gamma_t*dvct.x;
		ff_tan.y = kappa_t*zeta.y + gamma_t*dvct.y;
		ff_tan.z = kappa_t*zeta.z + gamma_t*dvct.z;
		aux = ff_tan.x*ff_tan.x + ff_tan.y*ff_tan.y + ff_tan.z*ff_tan.z;
		tangential = sqrt(aux);

		if (touchVec[idx].dynF)
		{
			tan_max = xmu*tan_max;
			if (tangential > tan_max)
			{
				aux = 1.0/tangential;
				ss.x = ff_tan.x*aux;
				ss.y = ff_tan.y*aux;
				ss.z = ff_tan.z*aux;

				ff_tan.x = tan_max*ss.x;
				ff_tan.y = tan_max*ss.y;
				ff_tan.z = tan_max*ss.z;

				aux = 1.0/kappa_t;
				zeta.x = (ff_tan.x - gamma_t*dvct.x)*aux;
				zeta.y = (ff_tan.y - gamma_t*dvct.y)*aux;
				zeta.z = (ff_tan.z - gamma_t*dvct.z)*aux;
				tangential = tan_max;
			}

			if (tangential < tan_max) touchVec[idx].dynF = 0;
		}
		else if (tangential > tan_max)
		{
			tan_max = xmu*tan_max;

			aux = 1.0/tangential;
			ss.x = ff_tan.x*aux;
			ss.y = ff_tan.y*aux;
			ss.z = ff_tan.z*aux;

			ff_tan.x = tan_max*ss.x;
			ff_tan.y = tan_max*ss.y;
			ff_tan.z = tan_max*ss.z;

			aux = 1.0/kappa_t;
			zeta.x = (ff_tan.x - gamma_t*dvct.x)*aux;
			zeta.y = (ff_tan.y - gamma_t*dvct.y)*aux;
			zeta.z = (ff_tan.z - gamma_t*dvct.z)*aux;
			tangential = tan_max;

			if (xmu != 1.0) touchVec[idx].dynF = 1;
		}

		touchVec[idx].zeta = zeta;
		touchVec[idx].nn = nn;
		break;
	}

	// Si no existe lo crea
	if (!found_flag) for (cc=0; cc<nTouch; cc++)
	{
		idx = cc + ind*nTouch;
		if (touchVec[idx].tag != -1) continue;

		found_flag = 1;
		touchVec[idx].flag = 1;
		touchVec[idx].tag = bb;

		// Actualiza
		zeta.x = dvct.x*dt;
		zeta.y = dvct.y*dt;
		zeta.z = dvct.z*dt;

		ff_tan.x = kappa_t*zeta.x + gamma_t*dvct.x;
		ff_tan.y = kappa_t*zeta.y + gamma_t*dvct.y;
		ff_tan.z = kappa_t*zeta.z + gamma_t*dvct.z;
		aux = ff_tan.x*ff_tan.x + ff_tan.y*ff_tan.y + ff_tan.z*ff_tan.z;
		tangential = sqrt(aux);

		if (tangential > tan_max)
		{
			tan_max = xmu*tan_max;

			aux = 1.0/tangential;
			ss.x = ff_tan.x*aux;
			ss.y = ff_tan.y*aux;
			ss.z = ff_tan.z*aux;

			ff_tan.x = tan_max*ss.x;
			ff_tan.y = tan_max*ss.y;
			ff_tan.z = tan_max*ss.z;

			aux = 1.0/kappa_t;
			zeta.x = (ff_tan.x - gamma_t*dvct.x)*aux;
			zeta.y = (ff_tan.y - gamma_t*dvct.y)*aux;
			zeta.z = (ff_tan.z - gamma_t*dvct.z)*aux;
			tangential = tan_max;

			if (xmu != 1.0) touchVec[idx].dynF = 1;
		}

		touchVec[idx].zeta = zeta;
		touchVec[idx].nn = nn;
		break;	
	}

	// Si hay mas contactos de los espcificados detiene todos los hilos
	if (!found_flag) asm("trap;");

	ff.x += ff_tan.x;
	ff.y += ff_tan.y;
	ff.z += ff_tan.z;

	//Calcula torque T = r x F
	tt.x = (rrc.y - rra.y)*ff.z - (rrc.z - rra.z)*ff.y;
	tt.y = (rrc.z - rra.z)*ff.x - (rrc.x - rra.x)*ff.z;
	tt.z = (rrc.x - rra.x)*ff.y - (rrc.y - rra.y)*ff.x;

	pff->x = ff.x;
	pff->y = ff.y;
	pff->z = ff.z;

	ptt->x = tt.x;
	ptt->y = tt.y;
	ptt->z = tt.z;

	return 1;
}

/*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*/

__global__ void getForces(double3 *rrVec, double3 *vvVec, double3 *wwVec,
			double3 *ffVec, double3 *ttVec, grain_prop *grainVec,
			touch *touchVec, long *nCntcVec, long *tagCntcVec,
			double3 *rrTolvVec, double3 *rrcCntcVec, double3 *ffcCntcVec,
			parameters pars)
{
	double3 rra, rrb, drr, rrc, vva, vvb, nn, ff, wwa, wwb, tt;
	long ngrains, nTouch, indCntc, tag_init, tag_end, bb;
	double kappa, gamma, mu, xk, xg, xmu,
		kappa_gg, gamma_gg, mu_gg,
		kappa_gw, gamma_gw, mu_gw,
		kappa_gp, gamma_gp, mu_gp;
	double siloWidth, h_siloThick, siloLid, rad_a, rad_b,
		sum_radii, sum_radii2, dist2, dist, comp, dt;
	int flag;

	ngrains = pars.ngrains;

	long ind = threadIdx.x + blockIdx.x*blockDim.x;

	if (ind < ngrains)
	{
		rra = rrVec[ind];
		if (rra.z < 0.0) return;

		vva = vvVec[ind];
		wwa = wwVec[ind];
		rad_a = grainVec[ind].rad;

		siloWidth = pars.siloWidth;
		h_siloThick = 0.5*pars.siloThick;
		siloLid = pars.siloInit + pars.siloHeight;

		kappa_gg = pars.kappa_gg;
		gamma_gg = pars.gamma_gg;
		mu_gg = pars.mu_gg;

		kappa_gw = pars.kappa_gw;
		gamma_gw = pars.gamma_gw;
		mu_gw = pars.mu_gw;

		kappa_gp = pars.kappa_gp;
		gamma_gp = pars.gamma_gp;
		mu_gp = pars.mu_gp;

		xk = pars.xk_tn;
		xg = pars.xg_tn;
		xmu = pars.xmu_ds;

		dt = pars.dt;

		nTouch = pars.nTouch;
		tag_init = nTouch*ind;
		tag_end = tag_init + nCntcVec[ind];

		for (indCntc=tag_init; indCntc<tag_end; indCntc++)
		{
			bb = tagCntcVec[indCntc];

			if (bb > ngrains)
			{
				kappa = kappa_gw;
				gamma = gamma_gw;
				mu = mu_gw;

				vvb.x = vvb.y = vvb.z = 0.0;
				wwb.x = wwb.y = wwb.z = 0.0;

				/*+*+*+*+* GRANO-PARED_IZQUIERDA +*+*+*+*/

				if (bb == ngrains + 1)
				{
					rrb.x = 0.0;
					rrb.y = rra.y;
					rrb.z = rra.z;

					dist = rra.x;
					if (dist >= rad_a) continue;
					comp = rad_a - dist;

					nn.x = -1.0;
					nn.y = 0.0;
					nn.z = 0.0;

					rrc.x = 0.0;
					rrc.y = rra.y;
					rrc.z = rra.z;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-PARED_DERECHA +*+*+*+*/

				if (bb == ngrains + 2)
				{
					rrb.x = siloWidth;
					rrb.y = rra.y;
					rrb.z = rra.z;

					dist = siloWidth - rra.x;
					if (dist >= rad_a) continue;
					comp = rad_a - dist;

					nn.x = 1.0;
					nn.y = 0.0;
					nn.z = 0.0;

					rrc.x = siloWidth;
					rrc.y = rra.y;
					rrc.z = rra.z;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-TOLVA_IZQUIERDA +*+*+*+*/

				if (bb == ngrains + 3)
				{
					// Pone en el origen de coordenadas
					rrb = rrTolvVec[ind];

					// Ahora checa distancias
					drr.x = rrb.x - rra.x;
					drr.y = rrb.y - rra.y;
					drr.z = rrb.z - rra.z;
					dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;
					dist = sqrt(dist2);
					if (dist >= rad_a) continue;

					comp = rad_a - dist;

					nn.x = drr.x/dist;
					nn.y = drr.y/dist;
					nn.z = drr.z/dist;

					rrc.x = rrb.x;
					rrc.y = rrb.y;
					rrc.z = rrb.z;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-TOLVA_DERECHA +*+*+*+*/

				if (bb == ngrains + 4)
				{
					rrb = rrTolvVec[ind];

					// Ahora checa distancias
					drr.x = rrb.x - rra.x;
					drr.y = rrb.y - rra.y;
					drr.z = rrb.z - rra.z;
					dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;
					dist = sqrt(dist2);
					if (dist >= rad_a) continue;

					comp = rad_a - dist;

					nn.x = drr.x/dist;
					nn.y = drr.y/dist;
					nn.z = drr.z/dist;

					rrc.x = rrb.x;
					rrc.y = rrb.y;
					rrc.z = rrb.z;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-TECHO +*+*+*+*/
				if (bb == ngrains + 7)
				{
					rrb.x = rra.x;
					rrb.y = rra.y;
					rrb.z = siloLid;

					dist = rrb.z - rra.z;
					if (dist >= rad_a) continue;
					comp = rad_a - dist;

					nn.x = 0.0;
					nn.y = 0.0;
					nn.z = 1.0;

					rrc.x = rra.x;
					rrc.y = rra.y;
					rrc.z = siloLid;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-PISO +*+*+*+*/
//				if (bb == ngrains + 8)
//				{
//					rrb.x = rra.x;
//					rrb.y = rra.y;
//					rrb.z = 0.0;

//					dist = rra.z;
//					if (dist >= rad_a) continue;
//					comp = rad_a - dist;

//					nn.x = 0.0;
//					nn.y = 0.0;
//					nn.z = -1.0;

//					rrc.x = rra.x;
//					rrc.y = rra.y;
//					rrc.z = 0.0;

//					goto FUERZA_TORQUE;
//				}

				kappa = kappa_gp;
				gamma = gamma_gp;
				mu = mu_gp;

				/*+*+*+*+* GRANO-PLANO_POSTERIOR +*+*+*+*/
				if (bb == ngrains + 5)
				{
					rrb.x = rra.x;
					rrb.y = h_siloThick;
					rrb.z = rra.z;

					dist = rrb.y - rra.y;
					if (dist >= rad_a) continue;
					comp = rad_a - dist;

					nn.x = 0.0;
					nn.y = 1.0;
					nn.z = 0.0;

					rrc.x = rra.x;
					rrc.y = h_siloThick;
					rrc.z = rra.z;

					goto FUERZA_TORQUE;
				}

				/*+*+*+*+* GRANO-PLANO_FRONTAL +*+*+*+*/
				if (bb == ngrains + 6)
				{
					rrb.x = rra.x;
					rrb.y = -h_siloThick;
					rrb.z = rra.z;

					dist = -(rrb.y - rra.y);
					if (dist >= rad_a) continue;
					comp = rad_a - dist;

					nn.x = 0.0;
					nn.y = -1.0;
					nn.z = 0.0;

					rrc.x = rra.x;
					rrc.y = -h_siloThick;
					rrc.z = rra.z;

					goto FUERZA_TORQUE;
				}
			}

			/*+*+*+*+*+*+*+*+*+* GRANO-GRANO +*+*+*+*+*+*+*+*+*/

			rrb = rrVec[bb];
			vvb = vvVec[bb];
			wwb = wwVec[bb];
			rad_b = grainVec[bb].rad;

			sum_radii = rad_a + rad_b;
			sum_radii2 = sum_radii*sum_radii;

			// Checa distancias
			drr.x = rrb.x - rra.x;
			drr.y = rrb.y - rra.y;
			drr.z = rrb.z - rra.z;
			dist2 = drr.x*drr.x + drr.y*drr.y + drr.z*drr.z;
			if (dist2 >= sum_radii2) continue;

			// Contacto
			dist = sqrt(dist2);
			comp = sum_radii - dist;

			nn.x = drr.x/dist;
			nn.y = drr.y/dist;
			nn.z = drr.z/dist;

			// Punto de contacto
			rrc.x = 0.5*(rra.x + rrb.x + (rad_a - rad_b)*nn.x);
			rrc.y = 0.5*(rra.y + rrb.y + (rad_a - rad_b)*nn.y);
			rrc.z = 0.5*(rra.z + rrb.z + (rad_a - rad_b)*nn.z);

			kappa = kappa_gg;
			gamma = gamma_gg;
			mu = mu_gg;

			FUERZA_TORQUE:

			flag = force_and_torque(rra, vva, wwa, rrb, vvb, wwb,
				rrc, nn, comp, kappa, gamma, mu, xk, xg, xmu,
				ind, nTouch, bb, dt, touchVec, &ff, &tt);

			if (!flag) continue;

			ffVec[ind].x += ff.x;
			ffVec[ind].y += ff.y;
			ffVec[ind].z += ff.z;

			ttVec[ind].x += tt.x;
			ttVec[ind].y += tt.y;
			ttVec[ind].z += tt.z;

			rrcCntcVec[indCntc].x = rra.x - rrb.x;
			rrcCntcVec[indCntc].y = rra.y - rrb.y;
			rrcCntcVec[indCntc].z = rra.z - rrb.z;
			ffcCntcVec[indCntc] = ff;
		}

	} 

	return;
}
