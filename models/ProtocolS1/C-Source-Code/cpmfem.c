#include "functions.h"

int main(void)
{
	int nrrdof, d;
	int *dofpos;
	double *f, *u;
	double **klocal;
	int *kcol;
	double *kval;
	NOD *pn;
	VOX *pv;
	int NRc,c,v;
	int *csize;
	int incr, startincr;

	/// INITIALIZE ///
   	srand(SEED); mt_init();
   	pv = init_voxels();
	pn = init_nodes();

	startincr = 0;
	if(startincr==0) {NRc = init_cells(pv);write_cells(pv,0); }
	else {NRc = read_cells(pv,startincr);}
	csize = calloc(NRc, sizeof(int)); for(c=0;c<NRc;c++) {csize[c]=0;}
	for(v=0;v<NV;v++) {if(pv[v].ctag) {csize[pv[v].ctag-1]++;}}

	set_forces(pn);
	set_restrictions(pn);

	// local K matrix
	klocal = set_klocal();

	// global K matrix:
	kcol = calloc(10*NDOF,sizeof(int));
	kval = calloc(10*NDOF,sizeof(double));
	assembly(kcol,kval,klocal,pv);
	dofpos = calloc(NDOF,sizeof(int));
	nrrdof = arrange_dofpos(dofpos,pn);
	reduce_K(kcol,kval,dofpos);

	/// START SIMULATION ///
	for(incr=startincr; incr<NRINC; incr++)
	{
		printf("\nSTART INCREMENT %d",incr);
		write_cells(pv,incr);


		cell_forces(pv,pn,csize,NRc);

		// FEA part // parts of this can go out the loop depending on what changes
		f=calloc(nrrdof,sizeof(double)); place_node_forces_in_f(pn,f);
		u=calloc(nrrdof,sizeof(double)); set_disp_of_prev_incr(pn,u);
		solvePCG(kcol,kval,u,f,nrrdof); disp_to_nodes(pn,u);
		free(u); free(f);

		if(incr%5==0)
		{
			//write_forces(pn,incr);
			//write_disps(pn,incr);
			write_pstrain(pv,pn,incr);
		}

		CPM_moves(pv,pn,csize);
	}

	/// END ///
	printf("\nSIMULATION FINISHED!");
	free(pv); free(pn); free(klocal); free(kcol); free(kval); free(dofpos);
	return 0;
}

