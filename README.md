# stellarmass_pca

This library re-implements the method of [Chen et al. (2012)](http://adsabs.harvard.edu/abs/2012MNRAS.421..314C) to infer star-formation histories from medium-resolution, medium-SNR optical/NIR spectra using PCA and a pre-computed set of SFH models. Specifically, the goal is to extract stellar masses in SDSS-IV/MaNGA spaxels.

## Overall Idea

As in Chen et al. (2012), many (tens of thousands) of model spectra are generated as a library (using [FSPS's python bindings](http://dan.iel.fm/python-fsps/current/)), with distributions of the following:
* a star-formation history (SFH), which is the sum of (i) an underlying, continuous SFR; (ii) a series of superimposed, stochastic burst; and (iii) possibly a randomly-placed, exponential truncation.
  * the continuous model has the functional form `SFR ~ exp(-g * (t - t_form) )`
    * formation time (`time_form`) is uniformly distributed in the range 1.5 - 13.5 Gyr
    * e-folding timescale for the underlying continuous model (`eftu`), whose inverse is distributed uniformly in the range 0 - 1 Gyr^-1
  * strength of the burst components (ii), relative to the continuous background (i), is controlled by `A`, whose elements are logarithmically distributed between .03 and 4.
  * the duration of the burst components (`dt_burst`) is uniformly-distributed in the range 30 - 300 Myr. Bursts do not occur preferentially at any lag after the onset of SF (set by `t_form`), but since they have a constant pdf through time, later formation leads to statistically fewer bursts.
    * Burst probabilities are set such that 15% of generated spectra have experienced a burst in the past 2 Gyr. This conveniently corresponds to a Poisson distribution D(t/tau), where tau is the age of the universe. That is, statistically, galaxies burst once.
  * the truncation occurs with an overall probability of 30%, at a random time `time_cut` (after `time_form`). Following the truncation, the SFR evolves as `SFR ~ exp(-(t-t_cut)/eftc)`, where `eftc` is the e-folding time of the cutoff, logarithmically-distributed in the range 10 Myr - 1 Gyr.
* metallicity, which is interpolated over the range available.
  * 95% of spectra are between 20% and 250% solar, and the remainder between 2% & 20% (both uniform)
  * the option exists to model a distribution of metallicities, but that option is not utilized or implemented
* dust extinction, modelled as two components `tau_V` and `mu`. `mu` expresses the fractional amount of `tau_V` that affects stellar populations older than 10 Myr
  * `tau_V` is normally-distributed over the range 0 - 6, with a peak at 1.2, and 68% of the total probability-mass less than 2
  * `mu` is normally-distributed with a peak at 0.3 and 68% of the probability-mass between .1 and 1
* velocity dispersion, `sigma`, uniformly-distributed in the range 50 - 400 km/s

Also computed and stored are:
* the full spectrum returned by FSPS
* D4000 & Hd_A indices, measured as SDSS does (see Table 1 in [Balogh et al. (1999)](http://adsabs.harvard.edu/abs/1999ApJ...527...54B))
* r-band luminosity-weighted age
* stellar-mass-weighted age
* i- and z-band stellar mass-to-light ratios
  * originally, this was computed for the redshift range 0 - 0.8, in steps of .05, but since all MaNGA galaxies are nearby, we just compute at `z = 0`

## Usage

The philosophy is that M* calculations should be carried out on a galaxy-by-galaxy basis. So you should be able to load a galaxy's DRP and DAP outputs, and calculate resolved stellar masses for all spaxels at the same time. We should also only have to compute our model spectra once, and just load them from disk thereafter.

The meat of the computation will be done in instances of the `StellarPop_PCA` and `MaNGA_deredshift` classes (in `find_pcs.py`). But some prep is needed first...

Start by initializing a library of model spectra:

    PCA = find_pcs.StellarPop_PCA(...)

I use the `from_YMC()` class method, which loads some pre-computed BC03 templates provided by Yanmei Chen. This method will eventually be exchanged for a cleaner method, once FSPS-C3K is available and we start generating natively high-res spectra. You can then run PCA on the models with `PCA.run_pca_models()`, specifying a value for `q`, if you like.

Then, pick a MaNGA galaxy (I'm using 7815-6104), and load its LOGCUBE and DAP-MAPS files into dereddening object, along with your CSP spectral characteristics:

    dered = find_pcs.MaNGA_deredshift(...)
    flux_regr, ivar_regr, mask_spax = dered.regrid_to_rest(
        template_logl=pca.logl, template_dlogl=pca.dlogl)
    mask_cube = dered.compute_eline_mask(
        template_logl=pca.logl, template_dlogl=pca.dlogl)

The `from_filenames()` class method will allow you to just specify a path to the LOGCUBE and MAPS files.

Finally, project the galaxies down onto the PCs.

    A = PCA.project_cube(
        f=flux_regr, ivar=ivar_regr,
        mask_spax=mask_spax, mask_cube=mask_cube)

`PCA.project_cube()` employs an weighted-chi-square-minimization approach to the projection, such that wavelength bins with high uncertainty are downweighted.
