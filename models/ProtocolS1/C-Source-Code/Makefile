OBJECTS = CPM_dH.o \
	FE_assembly.o \
	FE_local.o \
	FE_nodes2dofs.o \
	FE_solver.o \
	cellforces.o \
	cellmoves.o \
	init.o \
	mt.o \
	mylib.o \
	read.o \
	cpmfem.o \
	write.o

cpmfem: $(OBJECTS)
	cc -O -o cpmfem $(OBJECTS)	

clean:
	rm *.o

.c.o:
	cc -O -c $< -o $@
