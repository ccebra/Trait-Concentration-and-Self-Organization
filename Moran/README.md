## Evolution test for bimatrix games using Moran process.

Evolution_Test_moran_parallel.m is the evolution test for concentration we used for the bimatrix games process.

bimatrix_Moran_sampler.m carries out the Moran process until fixation on a grid of potential strategies for a given payout matrix.

Moran_performance_function_interp.m takes the output of bimatrix_Moran_sampler and interpolates it into a performance function. Note if performance is smooth enough this can instead be approximated by a polynomial equation (which is what we did with stag and PD).
