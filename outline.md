'''
I'm just writing a basic workflow for the example 2DTM
'''

### READ IN reference FILE ###
# This can be either a mrc map or a pdb/cif model
# If the model is supplied. Generate a 3D map

### OPTIONAL - MODIFY 3D MAP ###
# Here could apply: B factor, MTF, envelope functions (e.g. Cc),
# phase randomisation, and whitening filter, to the map.
# This could also be done in 2D when applying the CTF

### GENERATE SAMLPING GRID ###
# Generate a grid of angles using H3 or Hopf fibration algorithms
# Also z axis for defocus
# How fine the grid is and the range is user controlled.

### GENERATE REFERENCE PROJECTIONS FOR THE GRID ###
# 2D projections generated with a CTF applied
# potentially apply any other modifications
# Need to be careful about storage in doing this

### READ IN IMAGES ###
# How many at a time depends again on storage
# May be able to define relationships between images
# (e.g. related by known tilt angle or defocus)

### OPTIONAL - MODIFY IMAGES ###
# Can apply: whitening filter, phase randomisation,
# to the images.

### PERFORM CROSS CORRELATION ACROSS GRID ###
# Must have flexibility in how to run it (CPU/GPU cluster)
# As a first test just store the max for each pixel
# Later on allow refinement and storage of a distribution
# for a given set of pixels.
# Efficient memory usage will be key

### OUTPUT RESULTS ###
# I like just give the results and leave interpretation to user
# Could also allow definition of minimum target-distance and
# calculate some SNRs
