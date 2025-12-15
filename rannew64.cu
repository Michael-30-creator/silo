/* 
  function rannew64: este es un generador de numeros aleatorios que usa
  un fibonacci combinado con un congruencial. ambos se toman modulo 
  2^64, simplemente por overflow.
  funciona asi:
 
 	x(n) = x(n-NP) - x(n-NQ) mod 2^64 (via overflow)
  	y(n) = a*y(n-1) + c mod 2^64 (via overflow)
  	z(n) = x(n) - y(n) mod 2^64 (via overflow)

  si queremos devolver el idum se puede an~adir

  	idum(n) = z(n)/2 (usando >>)
 
  el periodo es (2^NP -  1)*2^(63) (aqui 63 es el numero de bits de 
  precision menos 1). se corre usando unsigned long.
  inicializa a la primera llamada y cada vez que tenga idum negativo. 
  en principio, debe generar (poniendo cuidado a la inicializacion) 
  2^((NP-1)(63)) ciclos disjuntos

  los valores de NP y NQ ("taps") que se pueden usar se toman de Coddington:

  NP	NQ

  17	5
  31	13
  55	24
  97	33
  127	63
  250	103
  521	168
  607	273
  1279	418
  2281	1029
  4423	2098
  9689	4187

  se supone que mientras mas grandes los taps, mejores los numeros. 
  (obviamente, mas largos los ciclos y mas ciclos).

  se inicializa con un congruencial de mod 2^64 
  distinto al que se usa en 
  el genreador.

  ultima modificacion: marzo 6 2009
*/


#define TRUE 1
#define FALSE 0

#define NP 250
#define NQ 103

double rannew64(long *pidum)
{
  static unsigned long useed[NP], uu, uux, uuy;
  static unsigned long ua = 2862933555777941757, uc = 3037000493;
  static double xmm = 18446744073709551615.;
  static double omm;
  static long idum;
  static int i, npp, not_init = TRUE;
  static int np = NP-1;
  static int nq = NQ-1;

/* corre inicializacion */

  idum = *pidum;
  if (not_init || (idum < 0))
  {
    unsigned long udum;
    /* unsigned long uaa = 69069, ucc = 1; */
    unsigned long uaa = 6364136223846793005, ucc = 1442695040888963407; 

/* arregla idum en caso de problema. fija otros parametros */

    if (idum < 0) 
    { 
      idum = -idum;
      *pidum = idum;
    }
    udum = (unsigned long) idum;

/* fill up the fibonacci vector with a congruential */

    for (i=0; i<1000; i++) udum = udum*uaa + ucc;
    for (i=0; i<NP; i++) 
    {
      udum = udum*uaa + ucc;
      useed[i] = udum;
    }

    npp = NP - 1; 
    omm = 1./xmm;
    uuy = udum;
    not_init = FALSE;
  }

/* esta es la parte que hace el fibo */

  uux = useed[np] - useed[nq];
  useed[np] = uux;
  np--;
  nq--;
  if (np < 0) np = npp;
  if (nq < 0) nq = npp;

/*ahora el congruencial*/

  uuy = ua*uuy + uc;

/*combina*/

  uu = uux - uuy;

/*si queremos devolver el idum, lo reduce a long. esto se
puede dejar comentado */

  /*
  uu = uu>>1;
  idum = (long) uu;
  *pidum = idum;
  */

  return (omm*((double) uu));
}
