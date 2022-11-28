Within this folder, we present the kinematic analysis of the [OIII]4959 and [OIII]5007 emission lines for the whole 19-type II quasars sample.

Each notebook follows a similar structure:
- For a start, we import all the needed packages and files with the predefined functions and parameters, followed by the information about the targets and the spectra 
of the studied target.
- Then, the kinematic analysis of the emission lines is performed, based on a parametric modeling of both emmision lines, followed by the non-parametric analysis of
the resulting modelled [OIII]5007 emission line. Also, the outflow component of each emission line, and its energetic properties, are estimated by using three different
methods.
-  First performed analysis is a multi-gaussian parametric model of both [OIII]4959 and [OIII]5007 emission line profiles, verifying a few relations between both 
doublet lines (defined in tied_parameters.py.)
- Then, a non-parametric analysis is executed for the [OIII]5007 emission line, using as input the previously modelled non-noisy profile.
- 

Besides each notebook, there are also three auxiliar files:
- aux_functions.py aux_functions_mask.py and present the predefined functions needed for the kinematic analysis proccess.
- tied_parameters.py collects the conditions for tying parameters between analogous gaussian components for the [OIII]4959 and [OIII]5007 emission lines.
