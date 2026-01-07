/* 
  function rannew64: this is a random number generator that uses
  a combined Fibonacci with a congruential. Both are taken modulo
  2^64, simply by overflow.
  It works as:
 
	x(n) = x(n-NP) - x(n-NQ) mod 2^64 (via overflow)
	y(n) = a*y(n-1) + c mod 2^64 (via overflow)
	z(n) = x(n) - y(n) mod 2^64 (via overflow)

  If we want to return idum we can add

	idum(n) = z(n)/2 (using >>)
 
  The period is (2^NP - 1)*2^63 (here 63 is the number of bits of
  precision minus 1). It runs using unsigned long.
  It initializes on the first call and each time idum is negative.
  In principle, it should generate (with careful initialization)
  2^((NP-1)(63)) disjoint cycles.

  The values of NP and NQ ("taps") that can be used are taken from Coddington:

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

  It is assumed that larger taps give better numbers.
  (obviously, longer cycles and more cycles).

  It initializes with a mod 2^64 congruential
  different from the one used in
  the generator.

  last modification: March 6 2009
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

/* run initialization */

  idum = *pidum;
  if (not_init || (idum < 0))
  {
    unsigned long udum;
    /* unsigned long uaa = 69069, ucc = 1; */
    unsigned long uaa = 6364136223846793005, ucc = 1442695040888963407; 

/* fix idum if needed. set other parameters */

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

/* this is the part that does the Fibonacci */

  uux = useed[np] - useed[nq];
  useed[np] = uux;
  np--;
  nq--;
  if (np < 0) np = npp;
  if (nq < 0) nq = npp;

/* now the congruential */

  uuy = ua*uuy + uc;

/* combine */

  uu = uux - uuy;

/* if we want to return idum, reduce to long. this can be
left commented */

  /*
  uu = uu>>1;
  idum = (long) uu;
  *pidum = idum;
  */

  return (omm*((double) uu));
}
