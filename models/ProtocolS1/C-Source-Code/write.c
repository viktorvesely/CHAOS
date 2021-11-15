// file: write.c
#include "functions.h"

////////////////////////////////////////////////////////////////////////////////
void write_increment(int increment)
{
	FILE *ofp;

	ofp = fopen("increment_number.out","w");
	fprintf(ofp,"%d",increment);
	fflush(ofp); fclose(ofp);
}


////////////////////////////////////////////////////////////////////////////////
void write_cells(VOX* pv, int increment)
{
	int v;
    int vx,vy;
   	char filename[20];
   	char astring[20];
   	FILE *ofp;

   	myitostr(increment, astring);
	strcpy(filename, "ctags");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");
	/*for(v=0; v<NV; v++)
		fprintf(ofp ,"%d\n", pv[v].ctag);*/
    for(vx=0; vx<NVX; vx++) {
        for (vy=0; vy<NVY; vy++) {
            v = vx + vy * NVX;
     fprintf(ofp ,"%d ", pv[v].ctag);
        }
        fprintf(ofp, "\n");
    }
   	fflush(ofp);  fclose(ofp);
}


/*
////////////////////////////////////////////////////////////////////////////////
void write_sed(VOX* pv, NOD* pn, int increment)
{
	int v, vx, vy;
	double sed;
   	char filename[20];
   	char astring[20];
   	FILE *ofp;

   	myitostr(increment, astring);
	strcpy(filename, "sed");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");

	for(vy=0; vy<NVY; vy++)
   	for(vx=0; vx<NVX; vx++)
   	{
   		v = vx + vy*NVX;
		sed = get_sed(vx,vy,pn);
		fprintf(ofp ,"%lf\n",sed);
	}

   	fflush(ofp); fclose(ofp);
}
*/

////////////////////////////////////////////////////////////////////////////////
void write_pstrain(VOX* pv, NOD* pn, int increment)
{
	int v;
	double estrains[3],L1,L2,v1[2],v2[2];
	char filename[20],filename2[20];
   	char astring[20];
   	FILE *ofp;

	myitostr(increment, astring);
	strcpy(filename, "pstrain");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");
	for(v=0;v<NV;v++)
	{
		get_estrains(pn,v,estrains);
		L1=L2=.0; get_princs(estrains,&L1,&L2,v1,v2,1);
		if(L1>L2)
		{
			fprintf(ofp,"%d ", (int)(1000000*L1));
			fprintf(ofp,"%d ", (int)(1000*v1[0]));
			fprintf(ofp,"%d ", (int)(1000*v1[1]));
			fprintf(ofp,"%d\n",(int)(1000000*L2));
		}
		else
		{
			fprintf(ofp,"%d ", (int)(1000000*L2));
			fprintf(ofp,"%d ", (int)(1000*v2[0]));
			fprintf(ofp,"%d ", (int)(1000*v2[1]));
			fprintf(ofp,"%d\n",(int)(1000000*L1));
		}
	}
	fflush(ofp); fclose(ofp);
}


/*
////////////////////////////////////////////////////////////////////////////////
void write_pstress(VOX* pv, NOD* pn, int increment)
{
	int v;
	double estrains[3],estress[3],L1,L2,v1[2],v2[2];
	char filename[20];
   	char astring[20];
   	FILE *ofp;

	myitostr(increment, astring);
	strcpy(filename, "pstress");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");
	for(v=0;v<NV;v++)
	{
		get_estrains(pn,v,estrains);
		get_estress(v,estrains,estress);
		L1=L2=.0; get_princs(estress,&L1,&L2,v1,v2,0);
		if(L1>L2)
			fprintf(ofp ,"%e\n",L1);
		else
			fprintf(ofp ,"%e\n",L2);
	}
	fflush(ofp); fclose(ofp);
}
*/

////////////////////////////////////////////////////////////////////////////////
void write_forces(NOD* pn, int increment)
{
	int n;
	char filename[20];
   	char astring[20];
	FILE *ofp;

	myitostr(increment, astring);
	strcpy(filename, "forces");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");
	for(n=0;n<NN;n++)
		fprintf(ofp ,"%e\n",pn[n].fx);
	for(n=0;n<NN;n++)
		fprintf(ofp ,"%e\n",pn[n].fy);
	fflush(ofp); fclose(ofp);
}


////////////////////////////////////////////////////////////////////////////////
void write_disps(NOD* pn, int increment)
{
	int n;
	char filename[20];
   	char astring[20];
	FILE *ofp;

	myitostr(increment, astring);
	strcpy(filename, "disps");
   	strcat(filename, astring);
   	strcat(filename, ".out");

	ofp = fopen(filename,"w");
	for(n=0;n<NN;n++)
		fprintf(ofp ,"%e\n",pn[n].ux);
	for(n=0;n<NN;n++)
		fprintf(ofp ,"%e\n",pn[n].uy);
	fflush(ofp); fclose(ofp);
}
