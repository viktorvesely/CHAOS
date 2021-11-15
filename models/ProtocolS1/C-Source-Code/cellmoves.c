// file: cellmoves.c
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////
void CPM_moves(VOX* pv, NOD* pn, int* csize)
// cellular potts model: one Monte Carlo step
{
	int i,NRsteps = NV;
	int xs, xt; // source and target pixel
	int xtx,xty; // x and y position of target pixel
	int ttag, stag; // target and source label
	int nbs[8],pick; // neighbors of target pixel
	BOOL go_on;
	double dH, prob;

	for(i=0;i<NRsteps;i++)
	{
		//xt = (rand()*NV/RAND_MAX); // pick random element
		xt = mt_random()%NV; // pick random element
		xty = xt/NVX; xtx = xt%NVX;

		if((xtx>0)&&(xtx<NVX-1)&&(xty>0)&&(xty<NVY-1)) // exclude outer rim
		{
			nbs[0]=xt-1+NVX; nbs[1]=xt+NVX; nbs[2]=xt+1+NVX;
			nbs[7]=xt-1;                    nbs[3]=xt+1;
			nbs[6]=xt-1-NVX; nbs[5]=xt-NVX; nbs[4]=xt+1-NVX;
			pick = mt_random()%8;
			xs = nbs[pick]; // pick random neighbor

			ttag = pv[xt].ctag;
			stag = pv[xs].ctag;

			go_on = 0;
			if(ttag!=stag) //don't bother if no difference
			{
        		go_on = 1;
        		if(ttag) // if a cell in xt (retracting)
				{
            		if (splitcheckCCR(pv,csize,xt,ttag))
                		go_on = 0;
            		if(csize[ttag-1]==1) // cell cannot disappear (constraint may be removed)
                		go_on = 0;
				}
			}

			if(go_on)
			{
        		dH = calcdH(pv,pn,csize,xt,xs,pick,ttag,stag);
        		prob = exp(-IMMOTILITY*dH);
        		if (prob>(rand()/(double)RAND_MAX))
				{
            		pv[xt].ctag = stag; // a move is made
            		if(ttag) {csize[ttag-1]--;}
            		if(stag) {csize[stag-1]++;}
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
BOOL splitcheckCCR(VOX* pv, int* csize, int xt, int ttag)
{
	BOOL split;
	int nbs[8],n,nb,prev,curr,in;
	int v, nrblue, nrgrey, startnb;
	int greys[csize[ttag-1]];
	short CCAlabels[NV];
	int i, nrgrey0, g, nbsg[8];

	nbs[0]=xt-1+NVX; nbs[1]=xt+NVX; nbs[2]=xt+1+NVX;
	nbs[7]=xt-1;                    nbs[3]=xt+1;
	nbs[6]=xt-1-NVX; nbs[5]=xt-NVX; nbs[4]=xt+1-NVX;

	prev = pv[nbs[7]].ctag; in = 0;
	for(n=0;n<8;n++)
	{
		curr = pv[nbs[n]].ctag;
		if((prev!=ttag)&&(curr==ttag))
			in++;
		prev = curr;
	}

	split = FALSE;
	if(in>1)
	{
		// CONNECTED COMPONENT ALGORITHM Rene-style (CCR)
    	// connected checking "label":
    	// 0: blue;    neighbors of retracted element
    	// 1: white;   undiscovered
    	// 2: grey;    discovered but not finished processing
    	// 3: black;   finished processing

		for(v=0;v<NV;v++) {CCAlabels[v] = 1;}
		CCAlabels[xt] = 3;

		nrblue = -1;
		for(n=0;n<8;n++)
		{
			nb = nbs[n];
			if(pv[nb].ctag==ttag)
			{
				CCAlabels[nb]=0; nrblue++;
				startnb = nb;
			}
		}
		CCAlabels[startnb]=2; nrgrey=1; greys[0]=startnb;

		while(nrgrey&&nrblue)
		{
			nrgrey0 = nrgrey;
			// make neighbors of discovered greys grey
			for(i=0;i<nrgrey0;i++)
			{
				g = greys[i];
				nbsg[0]=g-1+NVX; nbsg[1]=g+NVX; nbsg[2]=g+1+NVX;
				nbsg[7]=g-1;                    nbsg[3]=g+1;
				nbsg[6]=g-1-NVX; nbsg[5]=g-NVX; nbsg[4]=g+1-NVX;
				for(n=0;n<8;n++)
				{
					nb = nbsg[n];
					if((pv[nb].ctag==ttag)&&(CCAlabels[nb]<2))
					{
						if(CCAlabels[nb]==0) {nrblue--;}
						CCAlabels[nb]=2; nrgrey++; greys[nrgrey-1]=nb;
					}
				}
			}

			// make processed greys black
			for(i=0;i<nrgrey0;i++)
			{
				g = greys[i];
				CCAlabels[g]=3;
				greys[i]=greys[nrgrey-1]; nrgrey--;
			}

		}
		if(nrblue) {split = TRUE;}

	}
	return split;
}


