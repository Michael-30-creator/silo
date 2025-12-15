HOST_COMPILER = gcc
#HOST_COMPILER = clang-3.8
GPU_ARCH = sm_35
NVCCFLAGS = -O2
LIBRARIES = -lm
NVCC = nvcc -ccbin $(HOST_COMPILER) $(NVCCFLAGS) -arch=$(GPU_ARCH)

OBJS = update_verlet.o get_forces.o \
	get_contacts.o clean_and_collect.o rannew64.o siloTolva.o

siloTolva : $(OBJS)
	$(NVCC) $^ -o $@ $(LIBRARIES)

%.o : %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o *~

clean-all:
	rm -f *.o *~ *.dat bitacora *.xyz
