# What the data can and cannot tell you

The physics behind rules 1-2 and 15-18 of the skill: what a peak measures, what the instrument adds, and what a map
can therefore claim. The manual's Theory section derives each part.

## Microstructures: no smoothness, mixtures per voxel

anri targets everything from perfect crystals to heavily deformed metals. At voxels of 0.1-1 µm and rays tens to
hundreds of µm long:

- Annealed grains and twins: orientation constant to below the resolution, with sharp boundaries.
- Deformed FCC and BCC metals: dislocation cells of 0.5-2 µm with walls of 0.1-10° misorientation, accumulating like
  a random walk. Additively manufactured metals: solidification cells of 0.3-1 µm, ≲0.5° apart, drifting by degrees
  along a column.
- Martensite, bainite, Ti α laths: a few variants per voxel. Nano-twins: lamellae thinner than a voxel, so one voxel
  holds two orientations with a volume fraction.
- Smooth fields (elastic bending, elastic strain away from defects) are the exception.

So a voxel's orientation distribution is a small mixture of populations, each nearly constant with a small spread.
That is what the populations stage reports. anri couples no voxel to its neighbours: no smoothing, which would
suppress the local variation being measured.

## What a peak is

- A peak at one (dty, ω) sums every orientation the beam lights along the ray: a histogram along the ray, not one
  voxel's spot. No single image says which voxel made which part of it; only consistency across rows and across
  reflections places it.
- A rotation moves intensity along the Debye-Scherrer ring (η) and over frames (ω); strain moves it across the ring:
  Δ2θ = 2 ε tan θ. Sub-grain misorientation gives smeared, curved arcs (bananas), usually with a single maximum.
  Separate sub-spots are the exception. Don't mode-seek on a peak's maximum.
- A small lattice rotation δ = (δx, δy, δz) (beam along x, rotation axis z, η from vertical) moves a peak by
  Δω = −δz − δy cot η and Δη = δx − δy tan θ / sin η. An isotropic spread σ widens a peak by σ / |sin η| in ω:
  peaks near η = 0 and 180° (12 and 6 o'clock) smear most, which is why the indexer cuts them (`--etacut`).
- A rotation about g is invisible to that reflection, so each reflection sees only part of a spread. Spreads are
  anisotropic, and range from below 0.01° to about 10°.

## The instrument (ESRF ID11)

Most data come from scanning 3DXRD at ID11. Which optics, energy and overfocus a dataset used is not in its files:
ask.

- Optics: Al CRL boxes (f ~50 cm, effective aperture ~105-118 µm) or Si planar nanolenses (f ~10 cm, effective
  aperture ~38 µm). Convergence at the sample is at most about aperture / f: ~0.2 mrad for the CRLs, ~0.4-0.5 mrad
  for the nanolenses. A transfocator set to overfocus widens the beam at the sample by an amount chosen per
  experiment.
- The beam is Gaussian across dty; the dty step is usually its FWHM. The indexer assumes FWHM = dty step unless
  given `--beam`.
- Measured on undeformed Si with the CRLs at 0.05° frames: peak cores narrower than one frame at every η (~0.05°
  near η = 0), with weak tails over 5-9 frames (most likely thermal diffuse scattering, not extra populations). Omega
  widths beyond that are the sample's.
- What widens a peak in ω: horizontal convergence (the same at every η), vertical convergence and an axis tilt
  (|cot η|, zero at 3 and 9 o'clock), orientation spread (1/|sin η|), and the energy bandwidth ((Δλ/λ) tan θ /
  |sin η|, the only term growing with 2θ). An undeformed single crystal separates the instrument from the sample.
- The TDXRD station's rotation stage has ~1 µm radial runout: voxels shift between dty rows, periodically in ω. It
  does not widen peaks in ω.
- Frames integrate through motion: every frame over its ω step, and over its dty range where dty moves during the
  rotation (fly and helical scans; the histogram stage logs `dty moves within the rows`).
- The sparse data are thresholded by segmentation: the faint ends and tails of a peak can be missing. An empty bin is
  censored, not zero (`--censor`).
- Intensities can be normalised to a monitor counter (`--monitor`). Absorption along the incoming and diffracted
  paths can matter (the fit models it from `--cif` or `--mu`); extinction is ignored.

## Consequences for a report

- Populations closer than the resolution cannot be separated: report a mean and a spread, not invented structure.
- The shape of a peak carries the information; anri fits pixels (histogram bins), not centroids.
- Path averaging, where each voxel takes its rays' average, shrinks orientation shifts towards the grain mean.
  Check for it on a phantom with known shifts before claiming sub-grain variation.
