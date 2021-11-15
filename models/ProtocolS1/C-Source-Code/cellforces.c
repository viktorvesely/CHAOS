// file: cellforces.c
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////
void cell_forces(VOX* pv, NOD* pn, int* csize, int NRc)
{
	int c;
	int n,nx,ny;
	int v,vx,vy, cnttag;
	int NRcelln,cellnodes[NN];
	int i,j, n2;
	double dnx,dny,forcex,forcey;

	for(ny=1; ny<NNY-1; ny++)
   	for(nx=1; nx<NNX-1; nx++)
   	{
   		n = nx + ny*NNX;
		pn[n].fx = 0;
		pn[n].fy = 0;
	}

	for(c=0;c<NRc;c++)
	{
		// determine which nodes belong to cell c
		NRcelln = 0;
		for(ny=1; ny<NNY-1; ny++)
   		for(nx=1; nx<NNX-1; nx++)
   		{
   			n = nx + ny*NNX;
			cnttag = 0;
			for(vy=ny-1; vy<ny+1; vy++)
			for(vx=nx-1; vx<nx+1; vx++)
			{
				v = vx + vy*NVX;
				if(pv[v].ctag == c+1)
					cnttag++;
			}
			if(cnttag>0) // all cell nodes
			{
				cellnodes[NRcelln] = n;
				NRcelln++;
			}
		}

		// forces between cellnodes
		for(i=0;i<NRcelln;i++)
		{
			n = cellnodes[i];
			ny=n/NNX; nx=n%NNX;

			for(j=0;j<NRcelln;j++)
			{
				n2 = cellnodes[j];
				dny=(n2/NNX-ny)*VOXSIZE; // y distance between n and n2
				dnx=(n2%NNX-nx)*VOXSIZE; // x distance between n and n2

				forcex = CELLFORCE*dnx;
				forcey = CELLFORCE*dny;

				pn[n].fx += forcex;
				pn[n].fy += forcey;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
