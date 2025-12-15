#define PI 3.14159265358979323846
#define GRAV 981.0

typedef struct 
{ 
	double rad;
	double mass;
	double inertia;
} 
grain_prop;

typedef struct 
{ 
	double3 zeta;
	double3 nn;
	long tag;
	int flag;
	int dynF;
} 
touch;

typedef struct
{
	long ngrains;
	long nCell_x;
	long nCell_z;
	long nCell2;
	long nTags;
	long nTot;
	long nTouch;
	long nBinsHop;
	double siloWidth;
	double siloHeight;
	double siloThick;
	double siloInit;
	double hopWidth;
	double hopAngR;
	double hopLength;
	double gamma_gg;
	double kappa_gg;
	double mu_gg;
	double gamma_gw;
	double kappa_gw;
	double mu_gw;
	double gamma_gp;
	double kappa_gp;
	double mu_gp;
	double xk_tn;
	double xg_tn;
	double xmu_ds;
	double dt;
	double cellSide_x;
	double cellSide_z;
	double radMax;
	double binSize;
	double bottGap;
	double winWidth;
}
parameters;
