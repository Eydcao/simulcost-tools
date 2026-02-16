# Attached is a compressed folder with profiles (pSHOT.0TIME) and equilibria (gSHOT.0TIME) attached.
# In order to generate the corresponding cgyro quantities, you can do the following
# $profiles_gen -g $gfile -i $pfile
# $profiles_gen -i input.gacode -loc_rad $rmin

# This will generate input.locrad.cgyro at radius rmin (you can vary rmin from from 0.2 to 0.95, I would say)
# The number of cases is large enough to give you some sensible parameter range
# Please, do pay attention to run pfile for a corresponding gfile (i.e. at least for the same shot number, if not for the same time slice) 