# EPOCH Setup

Note that epoch requires a linux environment. To control particle order, you will need 3 copies of epoch each compiled for a specific particle-weighting order. 

1. Clone Epoch from the following repo: https://github.com/epochpic/epoch

2. You should now see an epoch directory. Navigate to epoch/epoch1d. You should see a Makefile. In the Makefile, you should see various compile flags which you can edit to change the particle-weighting order.

    * 2nd Order: Uncomment the "DEFINES += $(D)PARTICLE_SHAPE_TOPHAT" line
    * 3rd Order: No need to uncomment any lines, the default is 3rd order 
    * 5th Order: Uncomment the "DEFINES += $(D)PARTICLE_SHAPE_BSPLINE3" line

3. Once the Makefile is edited, compile the code (ex: make COMPILER=intel)

4. Once complied, copy the path to bin/epoch1d. In runners/epoch.py there are variables called path_epoch(nth)Order that define the paths of these binaries for 2nd/3rd/5th orders.

5. In runners/input.deck on line 40, there is a variable for "physics_table_location". These tables are located in epoch1d/src/physics_packages/TABLES. Replace the path in the input.deck file with your path (any path for the particle orders will work)

# Install GNU Fortran compiler and OpenMPI
sudo apt update
sudo apt install gfortran openmpi-bin libopenmpi-dev
# prepare epoch_bin in home
mkdir ~/epoch_bin
# Claude append this dir to both bash and zsh

# clone the repo
git clone --recursive https://github.com/Warwick-Plasma/epoch.git
cd epoch
cd epoch1d

# No need to modify, the defualt is 3rd order
make COMPILER=gfortran
cp bin/epoch1d ~/epoch_bin/3rd

# modify the make file accoring to 2nd order
# Claude How should I do it in cmd?

# Then build with GNU compiler instead, for 2nd order
make COMPILER=gfortran
cp bin/epoch1d ~/epoch_bin/2nd

# modify the make file accoring to 5th order
# Claude How should I do it in cmd?

# Then build with GNU compiler instead, for 2nd order
make COMPILER=gfortran
cp bin/epoch1d ~/epoch_bin/5th

# Claude, then you should run a cmp function on every 2 pairs between them to make sure they are different