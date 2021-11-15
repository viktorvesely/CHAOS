// file: init.c
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////
VOX* init_voxels(void)
{
	VOX* pv;
	int v, vx, vy;
	int i;

   	pv = calloc(NV, sizeof(VOX));

	// set voxel information
   	for(vy=0; vy<NVY; vy++)
   	for(vx=0; vx<NVX; vx++)
   	{
   		v = vx + vy*NVX;

		//pv[v].x = vx * VOXSIZE; pv[v].y = vy * VOXSIZE;
		pv[v].ctag = 0;
	}
	return pv;
}

////////////////////////////////////////////////////////////////////////////////
NOD* init_nodes(void)
{
	NOD* pn;
	int n, nx, ny;

   	pn = calloc(NN, sizeof(NOD));

	// set node information
   	for(ny=0; ny<NNY; ny++)
   	for(nx=0; nx<NNX; nx++)
   	{
   		n = nx + ny*NNX;

		//pn[n].x = nx * VOXSIZE; pn[n].y = ny * VOXSIZE;
		pn[n].fx = .0; pn[n].fy = .0;
		pn[n].ux = .0; pn[n].uy = .0;
		pn[n].restrictx = FALSE; pn[n].restricty = FALSE;
	}
	return pn;
}

////////////////////////////////////////////////////////////////////////////////
int init_cells(VOX* pv)
{
	int v, vx, vy;
	int NRc;
	double r01;
	double d; int dx, dy; // distance to center

	NRc = 0;
	for(vy=0; vy<NVY; vy++)
   	for(vx=0; vx<NVX; vx++)
   	{
   		v = vx + vy*NVX;

		if((vx>0)&&(vx<NVX-1)&&(vy>0)&&(vy<NVY-1)) // exclude outer rim
		{
			r01 = rand()/(double)RAND_MAX;

			if(r01<.25/TARGETVOLUME)
			//if((vx==NVX/2)&&(vy==NVY/2))
			//if(((vx==NVX/2-7)||(vx==NVX/2+7))&&(vy==NVY/2))
			//dx=vx-NVX/2; dy=vy-NVY/2; d=sqrt(dx*dx+dy*dy); if((d<NVX/8.0) && (r01<1.5/TARGETVOLUME))
			{
				NRc++;
				pv[v].ctag = NRc;
               
			}
		}

	}

	return NRc;
}





////////////////////////////////////////////////////////////////////////////////
void set_forces(NOD* pn)
{
   	int n, nx, ny;
	double

	a = (0.0/6.0) * 3.1416;

	for(n=0; n<NN; n++)
	{
		pn[n].fx = .0;
		pn[n].fy = .0;
	}

   	for(ny=0; ny<NNY; ny++)
   	for(nx=0; nx<NNX; nx++)
   	{
   		n = nx + ny*NNX;

		// lower plate (iy==0) loading
		if(ny==0)
		{
        	pn[n].fx +=  sin(a)*cos(a)*FORCE;
			pn[n].fy += -cos(a)*cos(a)*FORCE;
       	}
		// upper plate (iy==NNY-1) loading
		if(ny==NNY-1)
       	{
        	pn[n].fx += -sin(a)*cos(a)*FORCE;
			pn[n].fy +=  cos(a)*cos(a)*FORCE;
		}
		// left plate (ix==0) loading
		if(nx==0)
		{
        	pn[n].fx += -sin(a)*sin(a)*FORCE;
			pn[n].fy +=  sin(a)*cos(a)*FORCE;
		}
		// right plate (ix==NNX-1) loading
		if(nx==NNX-1)
		{
        	pn[n].fx +=  sin(a)*sin(a)*FORCE;
			pn[n].fy += -sin(a)*cos(a)*FORCE;
        }
	}

   	for(ny=0; ny<NNY; ny++)
   	for(nx=0; nx<NNX; nx++)
   	{
   		n = nx + ny*NNX;

      	// for loading on the side of a plate, forces are lower
		if(((nx==0)||(nx==NNX-1)) && ((ny==0)||(ny==NNY-1)))
		{
			pn[n].fx *= .5;
         	pn[n].fy *= .5;
		}
   	}
}

////////////////////////////////////////////////////////////////////////////////
void set_restrictions(NOD* pn)
{

	/*
	int fixn1, fixn2, fixn3;

	// impose boundary conditions
	fixn1=0;
	fixn2=NNX-1;
	//fixn3=NNX*(NNY-1);


	pn[fixn1].restrictx=TRUE;
	pn[fixn1].restricty=TRUE;
	pn[fixn2].restricty=TRUE;
	//pn[fixn3].restrictx=TRUE;
	*/

	int n, nx, ny;

	for(ny=0; ny<NNY; ny++)
   	for(nx=0; nx<NNX; nx++)
   	{
   		n = nx + ny*NNX;

		if((nx==0)||(nx==NNX-1)||(ny==0)||(ny==NNY-1))
		{
			pn[n].restrictx=TRUE;
			pn[n].restricty=TRUE;
		}
	}

}

