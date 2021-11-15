// file: functions.h
#include "def.h"
#include "structures.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// init.c
VOX*		init_voxels(void);
NOD*		init_nodes(void);
int 		init_cells(VOX* pv);
void 		set_forces(NOD* pn);
void		set_restrictions(NOD* pn);

// cellmoves.c
void 		CPM_moves(VOX* pv, NOD* pn, int* csize);
BOOL 		splitcheckCCR(VOX* pv,  int* csize, int xt, int ttag);

// CPM_dH.c
double 		calcdH(VOX* pv, NOD* pn, int* csize, int xt, int xs, int pick, int ttag, int stag);
double 		calcdHcontact(VOX* pv, int xt, int ttag, int stag);
double 		contactenergy(int tag1, int tag2);
double 		calcdHvol(int* csize, int ttag, int stag);
double 		calcdHstrain(NOD* pn, int xt, int xs, int pick, int ttag, int stag);
double 		sige(double L);


// cellforces.c
void 		cell_forces(VOX* pv, NOD* pn, int* csize, int NRc);


// FE_local.c
double** 	set_klocal(void);
void 		material_matrix(double *pD);
void 		set_matrix_B(double *pB, double x, double y);
void 		get_estrains(NOD* pn, int e, double* estrains);
void 		get_estress(int e, double* estrains, double* estress);
void 		get_princs(double* str, double* L1, double* L2, double* v1, double* v2, BOOL strain);

// FE_assembly.c
//int** 		set_topology(void);
void		assembly(int* kcol, double* kval, double** klocal, VOX* pv);
int 		arrange_dofpos(int* dofpos, NOD* pn);
void 		reduce_K(int* kcol, double* kval, int* dofpos);

// FE_nodes2dofs.c
void 		disp_to_nodes(NOD* pn, double* u);
void 		set_disp_of_prev_incr(NOD* pn, double* u);
void 		place_node_forces_in_f(NOD* pn, double* f);

// FE_solver.c
void 		calc_Kdotx(int* kcol, double* kval, double* diag, double* x, double* b, int nrrdof);
void 		solvePCG(int* kcol, double* kval, double* u, double* f, int nrrdof);

// write.c
void   		write_increment(int increment);
void 		write_cells(VOX* pv, int increment);
void 		write_pstrain(VOX* pv, NOD* pn, int increment);
void 		write_pstress(VOX* pv, NOD* pn, int increment);
void 		write_forces(NOD* pn, int increment);
void 		write_disps(NOD* pn, int increment);

// read.c
int   		read_increment(void);
int 		read_cells(VOX* pv, int increment);

// mylib.c
void 		myitostr(int n, char s[]);
void 		myreverse(char s[]);
unsigned 	mystrlen(const char *s);

// mt.c
void 		mt_init();
unsigned long mt_random();


