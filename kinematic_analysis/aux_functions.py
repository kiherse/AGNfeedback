import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('tab10',8)
sns.set_style('whitegrid')
from numpy import random as rnd
from astropy.modeling import Model, models, fitting
import astropy.units as u
from scipy.interpolate import interp1d
import statsmodels.robust.scale as stats
from sklearn.metrics import auc

import copy

cspeed = 2.999e5 * u.km / u.s
speed = cspeed.value

def fit_continuum(wave,flux,wlimits):

    mask_cont = None
    for ilim,wlim in enumerate(wlimits):
        if ilim > 0 :
           mask_cont += ((wave >= wlim[0]) & (wave <= wlim[1]))
        else :
           mask_cont = ((wave >= wlim[0]) & (wave <= wlim[1])) 

    wave_cont = wave[mask_cont]
    flux_cont = flux[mask_cont]

    linfitter = fitting.LinearLSQFitter()
    poly_cont = linfitter(models.Polynomial1D(1), wave_cont, flux_cont)
    flux_mod_cont = poly_cont(wave_cont)

    resid_cont = flux_cont - flux_mod_cont
    noise = np.std(resid_cont)

    return poly_cont,noise

def write_output_fit(wave_reg,cont_reg_model,line_combo_fit,line_combo_error,\
    flux_norm=1.,logfile=None,verbose=True):

    if (logfile != None):
        f=open(logfile,mode='w')
    if line_combo_fit.n_submodels > 1:
        tab_row = []
        for submod,submod_error in zip(line_combo_fit,line_combo_error):
            #print(submod.name,submod.amplitude.value,submod.mean.value,submod.stddev.value)
            #print("Errors ",submod.name,submod_error.amplitude.value,submod_error.mean.value,submod_error.stddev.value)
            submod_flux, submod_flux_error = calc_gaussian_flux(submod,flux_norm=flux_norm,Errpars=submod_error)
            submod_fwhm,submod_fwhm_error=calc_gaussian_fwhm(submod,velocity=True,Errpars=submod_error)
            submod_cont = cont_reg_model[np.argmin(np.abs(wave_reg-submod.mean))]
            submod_eqw=submod_flux/(submod_cont*flux_norm) 
            submod_eqw_error=submod_flux_error/(submod_cont*flux_norm) 
            tab_row.append((submod.name,submod.mean,submod_error.mean,\
                submod_flux,submod_flux_error,submod_fwhm,submod_fwhm_error,\
                submod_eqw,submod_eqw_error))
            if verbose:
                print(submod.name)
                print(submod_flux,submod_cont,flux_norm)
                print(f' Center= {submod.mean.value:.2f} +/- {submod_error.mean.value:.2f}')
                print(f' FWHM= {submod_fwhm:.3e} +/- {submod_fwhm_error:.3e}')
                print(f' Flux= {submod_flux:.3e} +/- {submod_flux_error:.3e}')
                print(f' EW= {submod_eqw:.2f} +/- {submod_eqw_error:.2f}')
                print(" ")
            if (logfile != None):
                f.write(submod.name + '\n')
                f.write(f' Center= {submod.mean.value:.2f} +/- {submod_error.mean.value:.2f} \n')
                f.write(f' FWHM= {submod_fwhm:.3e} +/- {submod_fwhm_error:.3e} \n')
                f.write(f' Flux= {submod_flux:.3e} +/- {submod_flux_error:.3e} \n')
                f.write(f' EW= {submod_eqw:.2f} +/- {submod_eqw_error:.2f} \n')
                
    else:
        submod,submod_error = (line_combo_fit,line_combo_error)
        #print(submod.name,submod.amplitude.value,submod.mean.value,submod.stddev.value)
        #print("Errors ",submod.name,submod_error.amplitude.value,submod_error.mean.value,submod_error.stddev.value)
        submod_flux, submod_flux_error = calc_gaussian_flux(submod,flux_norm=flux_norm,Errpars=submod_error)
        submod_fwhm,submod_fwhm_error=calc_gaussian_fwhm(submod,velocity=True,Errpars=submod_error)
        submod_cont = cont_reg_model[np.argmin(np.abs(wave_reg-submod.mean))]
        submod_eqw=submod_flux/(submod_cont*flux_norm) 
        submod_eqw_error=submod_flux_error/(submod_cont*flux_norm)
        ## tab_row name,center,ecenter,flux,eflux,fwhm,efwhm,eqw,eeqw 
        tab_row=(submod.name,submod.mean,submod_error.mean,\
            submod_flux,submod_flux_error,submod_fwhm,submod_fwhm_error,\
            submod_eqw,submod_eqw_error)
        if verbose:
            print(submod.name)
            print(f' Center= {submod.mean.value:.2f} +/- {submod_error.mean.value:.2f}')
            print(f' FWHM= {submod_fwhm:.3e} +/- {submod_fwhm_error:.3e}')
            print(f' Flux= {submod_flux:.3e} +/- {submod_flux_error:.3e}')
            print(f' EW= {submod_eqw:.2f} +/- {submod_eqw_error:.2f}')
            print(" ")
        if (logfile != None):
            f.write(submod.name + '\n')
            f.write(f' Center= {submod.mean.value:.2f} +/- {submod_error.mean.value:.2f} \n')
            f.write(f' FWHM= {submod_fwhm:.3e} +/- {submod_fwhm_error:.3e} \n')
            f.write(f' Flux= {submod_flux:.3e} +/- {submod_flux_error:.3e} \n')
            f.write(f' EW= {submod_eqw:.2f} +/- {submod_eqw_error:.2f} \n')

    if (logfile != None):
        f.close()

    return tab_row

def perform_fit_plot(wave_reg,flux_reg,cont_reg_model,wave_O3r,wave_O3b,\
              line_combo,noise,Nsimul=1000,flux_norm=1.,plotfile=None,logfile=None,tgt_name=None,\
          plot_cont=True,tab_entry=None,plot_subtracted=False):

    flux_reg_nocont=flux_reg - cont_reg_model
    line_combo_fit,line_combo_error, nonpar_fit, nonpar_error, bestfit_outflow, errorfit_outflow,  bestfit_outflow_blue, errorfit_outflow_blue, bestfit_outflow_red, errorfit_outflow_red, params_simul, params_simul_nonpar = calc_bestfit_parameters(line_combo,wave_reg, \
                flux_reg_nocont,flux_norm,Noise=noise,Nsimul=Nsimul)
                
    blueshift,eblueshift,centroid_5007,centroid_4959 = Blueshift(line_combo_fit,line_combo_error)
    
    line_combo_mod = line_combo_fit(wave_reg) 

    tab_entry = write_output_fit(wave_reg,cont_reg_model,line_combo_fit,line_combo_error,\
        flux_norm=flux_norm,logfile=logfile)
    
    if (plotfile != None):
        xlabel=r"Wavelength [$\AA$] (Observed)"
        ylabel1='$F_\lambda$ $[10^{-15}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
        ylabel2=r'$F_\lambda-F_\lambda^{mod} $'
        norm_plot = 1.0e-15
        chi2=np.sum((flux_reg_nocont-line_combo_mod)**2/noise**2)
        f, (ax1, ax2) = plt.subplots(2, 1,figsize=(11,7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)    
        ax1.step(wave_reg,flux_reg*flux_norm/norm_plot,label='Observed',where='mid',color='gray',linewidth=3)
        ax1.plot(wave_reg,(cont_reg_model+line_combo_mod)*flux_norm/norm_plot,label='Full Model',linewidth=3)
        ax1.plot(wave_reg,cont_reg_model*flux_norm/norm_plot,color='burlywood',label='Stellar Model',linewidth=2)
        ax1.axvline(centroid_4959/u.Angstrom,linestyle='--',color='gray')
        ax1.axvline(centroid_5007/u.Angstrom,linestyle='--',color='gray')
        base_cont = np.median(cont_reg_model)*flux_norm/norm_plot*0.7
        if (plot_subtracted):
            ax1.step(wave_reg,(flux_reg-cont_reg_model)*flux_norm/norm_plot+base_cont,\
                label='Stellar sub',linewidth=1,where='mid')
        if line_combo_fit.n_submodels > 1:
            for submod_fit in line_combo_fit:
                ax1.plot(wave_reg,submod_fit(wave_reg)*flux_norm/norm_plot+base_cont,label=submod_fit.name,\
                    linestyle='--')
        else:        
            ax1.plot(wave_reg,line_combo_mod*flux_norm/norm_plot+base_cont,label=line_combo_fit.name,\
                linestyle='--')    
        ax1.legend(fontsize=14)
        ax1.set_ylabel(ylabel1,fontsize=16)
        ax1.tick_params(axis='both', labelsize=12)
        
        sns.set_palette('tab10',8)
        ax2.axhline(0,linestyle='--',color="gray")
        ax2.step(wave_reg,(flux_reg_nocont-line_combo_mod)*flux_norm/norm_plot,label='Residual',where='mid')
        ax2.axvspan(6863,6913,color='green',alpha=0.2)
        ax2.axvspan(71250,7326,color='yellow',alpha=0.2)
        ax2.set_xlabel(xlabel,fontsize=16)
        ax2.set_ylabel(ylabel2,fontsize=16)
        ax2.set_xlim(np.float(np.min(wave_reg)/u.Angstrom),np.float(np.max(wave_reg)/u.Angstrom))
        ax2.tick_params(axis='both', labelsize=12)
        f.subplots_adjust(hspace=0)
        f.suptitle(tgt_name,y=0.92,fontsize=18)
        f.savefig(plotfile,bbox_inches='tight')

    return line_combo_fit, line_combo_error, tab_entry, chi2, blueshift, eblueshift, nonpar_fit, nonpar_error, bestfit_outflow, errorfit_outflow,  bestfit_outflow_blue, errorfit_outflow_blue, bestfit_outflow_red, errorfit_outflow_red, params_simul, params_simul_nonpar

def indexes_gauss_comp(compoundmodel,icomp):
    icomp = 1
    param_names = ['amplitude_'+str(icomp),'mean_'+str(icomp),'stddev_'+str(icomp)]
    index_names =[]
    for name in param_names:
        index_names.append(compoundmodel.param_names.index(name))
    return index_names

def assign_errors(gauss_err):
    err_ndarray= np.array((gauss_err[0],gauss_err[1],gauss_err[2]),\
        dtype=[('amplitude_err','<f8'),('mean_err','<f8'),('stddev_err','<f8')])
    return err_ndarray

def calc_gaussian_fwhm(gaussian,velocity=True,Errpars=None):
    #print(gaussian)
    ## it should be done first the assignment and then numpy operations, otherwise 
    ## units are lost
    fwhm = gaussian.stddev
    fwhm *= np.sqrt(8*np.log(2.)) 
    if velocity:
        fwhm *= (cspeed/gaussian.mean)
    if Errpars:
        fwhm_err = Errpars.stddev
        fwhm_err *= np.sqrt(8*np.log(2.))
        if velocity:
            fwhm_err *= (cspeed/gaussian.mean)
        return fwhm, fwhm_err
    return fwhm

def calc_gaussian_flux(gaussian,flux_norm=1,Errpars=None):
    flux = np.sqrt(2*np.pi)*gaussian.stddev*flux_norm*gaussian.amplitude
    if Errpars:
        flux_err = np.sqrt(2*np.pi)*flux_norm*\
            np.sqrt(gaussian.stddev**2*Errpars.amplitude**2+\
            gaussian.amplitude**2*Errpars.stddev**2)
        return flux, flux_err   
    return flux

def calc_bestfit_parameters(fit_model,x,y,flux_norm,Noise=None,Nsimul=1000,verbose=False):
    
    # Select the fitter
    fitter = fitting.LevMarLSQFitter()
    
    # Select original data and initial fit
    bestfit_pars_0 = fitter(fit_model, x,y)
    bestfit_pars_line = None
    
    if all('4959' in c.name for c in bestfit_pars_0)==True:
        for c in bestfit_pars_0:
            if '4959' in c.name:
                if isinstance(bestfit_pars_line,Model):
                    bestfit_pars_line = bestfit_pars_line + c
                else:
                    bestfit_pars_line = c
                    
    else:
        for c in bestfit_pars_0:
            if '5007' in c.name:
                if isinstance(bestfit_pars_line,Model):
                    bestfit_pars_line = bestfit_pars_line + c
                else:
                    bestfit_pars_line = c
                
    x_ext =  np.linspace(min(x),max(x),1000)
    bestfit_mod_0 = bestfit_pars_line(x_ext)
    nonpar_fit_0 = nonpar_velocity(x_ext,bestfit_mod_0)
    bestfit_nonpar_0 = nonpar_fit_0.bestfit_nonpar()
    
    bestfit_outflow_0 = outflow(x_ext,bestfit_pars_line,flux_norm) # Giovanna method
    bestfit_outflow_blue_0,bestfit_outflow_red_0 = outflow_nonpar(x_ext,bestfit_pars_line,bestfit_nonpar_0,flux_norm) # Nonpar method
    
    if (Noise != None):
        
        print('\n')
        print('Initial parameters of the parametric model',bestfit_pars_0.parameters)
        print('\n')
        print('Initial velocities of the non-parametric analysis',bestfit_nonpar_0)
        print('\n')        
        
        # Set the arrays and functions for saving the simulated data
        bestfit_pars = copy.deepcopy(bestfit_pars_0)
        errorfit_params = copy.deepcopy(bestfit_pars_0)
        nparam = bestfit_pars_0.param_sets.size
        nparam_nonpar = len(bestfit_nonpar_0)
        nparam_outflow = len(bestfit_outflow_0)
        nparam_outflow_nonpar = len(bestfit_outflow_blue_0)
        params_simul = np.zeros([nparam,Nsimul])
        params_simul_nonpar = np.zeros([nparam_nonpar,Nsimul])
        params_simul_outflow = np.zeros([nparam_outflow,Nsimul])
        params_simul_outflow_blue = np.zeros([nparam_outflow_nonpar,Nsimul])
        params_simul_outflow_red = np.zeros([nparam_outflow_nonpar,Nsimul])
        
        # MonteCarlo: generate noisy spectra and repeat the fit
        print(' -------- ')
        print('\n')
        print(' Executing simulation')
        
        for isimul in range(Nsimul):
            
            # Random spectra
            y_sim = rnd.normal(y,Noise)
            
            # Parametric model of the random spectra
            bestfit_pars_sim = fitter(fit_model,x,y_sim)
            params_simul[:,isimul] = bestfit_pars_sim.parameters
            
            bestfit_pars_sim_line = None

            if all('4959' in c.name for c in bestfit_pars_sim)==True:
                for c in bestfit_pars_sim:
                    if '4959' in c.name:
                        if isinstance(bestfit_pars_sim_line,Model):
                            bestfit_pars_sim_line = bestfit_pars_sim_line + c
                        else:
                            bestfit_pars_sim_line = c

            else:
                for c in bestfit_pars_sim:
                    if '5007' in c.name:
                        if isinstance(bestfit_pars_sim_line,Model):
                            bestfit_pars_sim_line = bestfit_pars_sim_line + c
                        else:
                            bestfit_pars_sim_line = c
                        
            bestfit_mod_sim = bestfit_pars_sim_line(x_ext)
            
            # Non-parametric analysis of the random spectra
            nonpar_fit_sim = nonpar_velocity(x_ext,bestfit_mod_sim)
            params_simul_nonpar[:,isimul] = nonpar_fit_sim.bestfit_nonpar()
            
            # Outflow line flux and velocity
            params_simul_outflow[:,isimul] = outflow(x_ext,bestfit_pars_sim_line,flux_norm) #Giovanna
            params_simul_outflow_blue[:,isimul],params_simul_outflow_red[:,isimul]= outflow_nonpar(x_ext,bestfit_pars_sim_line,nonpar_fit_sim.bestfit_nonpar(),flux_norm) #Nonpar
            
        # Best-fit models from the simulation and errors
        # Parametric model
        bestfit_pars.parameters = np.median(params_simul,axis=1)
        errorfit_params.parameters = stats.mad(params_simul,axis=1)
        errorfit_params_mean = np.median(params_simul,axis=1)
        # Non-parametric analysis
        bestfit_nonpar = np.median(params_simul_nonpar,axis=1)
        errorfit_nonpar = stats.mad(params_simul_nonpar,axis=1)
        errorfit_nonpar_mean = np.median(params_simul_nonpar,axis=1)
        # Outflow properties
        bestfit_outflow = np.median(params_simul_outflow,axis=1)
        errorfit_outflow = stats.mad(params_simul_outflow,axis=1)
        
        bestfit_outflow_blue = np.median(params_simul_outflow_blue,axis=1)
        errorfit_outflow_blue = stats.mad(params_simul_outflow_blue,axis=1)
        bestfit_outflow_red = np.median(params_simul_outflow_red,axis=1)
        errorfit_outflow_red = stats.mad(params_simul_outflow_red,axis=1)

        return bestfit_pars, errorfit_params, bestfit_nonpar, errorfit_nonpar, bestfit_outflow, errorfit_outflow, bestfit_outflow_blue, errorfit_outflow_blue, bestfit_outflow_red, errorfit_outflow_red, params_simul, params_simul_nonpar

    else:
        return bestfit_pars

        errorfit_params.parameters=np.std(params_simul,axis=1) 

        return bestfit_pars, error_pars

def Blueshift(line_combo_fit,line_combo_error):

    blueshift=np.zeros(line_combo_fit.n_submodels)
    eblueshift=np.zeros(line_combo_fit.n_submodels)

    Centroid_5007=0*u.angstrom
    eCentroid_5007=0*u.angstrom
    Centroid_4959=0*u.angstrom
    eCentroid_4959=0*u.angstrom	

    for i in range(line_combo_fit.n_submodels):

        if line_combo_fit[i].name=='[OIII]5007n':
            Centroid_5007=line_combo_fit[i].mean
            eCentroid_5007=line_combo_error[i].mean

        if line_combo_fit[i].name=='[OIII]4959n':
            Centroid_4959=line_combo_fit[i].mean
            eCentroid_4959=line_combo_error[i].mean

        if line_combo_fit[i].name=='[OIII]5007n1':
            for j in range(line_combo_fit.n_submodels):
                if line_combo_fit[j].name=='[OIII]5007n2':
                    Centroid_5007=(line_combo_fit[i].mean*line_combo_fit[i].amplitude+line_combo_fit[j].mean*line_combo_fit[j].amplitude)/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude)
                    eCentroid_5007=np.sqrt((line_combo_fit[i].amplitude*line_combo_error[i].mean)**2+(line_combo_fit[j].amplitude*line_combo_error[j].mean)**2+((line_combo_fit[i].mean-line_combo_fit[j].mean)*line_combo_fit[j].amplitude*line_combo_error[i].amplitude/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude))**2+((line_combo_fit[j].mean-line_combo_fit[i].mean)*line_combo_fit[i].amplitude*line_combo_error[j].amplitude/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude))**2)/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude)

        if line_combo_fit[i].name=='[OIII]4959n1':
            for j in range(line_combo_fit.n_submodels):
                if line_combo_fit[j].name=='[OIII]4959n2':
                    Centroid_4959=(line_combo_fit[i].mean*line_combo_fit[i].amplitude+line_combo_fit[j].mean*line_combo_fit[j].amplitude)/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude)
                    eCentroid_4959=np.sqrt((line_combo_fit[i].amplitude*line_combo_error[i].mean)**2+(line_combo_fit[j].amplitude*line_combo_error[j].mean)**2+((line_combo_fit[i].mean-line_combo_fit[j].mean)*line_combo_fit[j].amplitude*line_combo_error[i].amplitude/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude))**2+((line_combo_fit[j].mean-line_combo_fit[i].mean)*line_combo_fit[i].amplitude*line_combo_error[j].amplitude/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude))**2)/(line_combo_fit[i].amplitude+line_combo_fit[j].amplitude)

    for m in range(line_combo_fit.n_submodels):		

        if line_combo_fit[m].name[:10]=='[OIII]5007':
            blueshift[m]=2.999e5*(line_combo_fit[m].mean/Centroid_5007-1)
            eblueshift[m]=2.999e5*(line_combo_error[m].mean/Centroid_5007)

        if line_combo_fit[m].name[:10]=='[OIII]4959':
            blueshift[m]=2.999e5*(line_combo_fit[m].mean/Centroid_4959-1)
            eblueshift[m]=2.999e5*(line_combo_error[m].mean/Centroid_4959)

    return blueshift, eblueshift, Centroid_5007, Centroid_4959

class nonpar_velocity:
    # measure line profile based on non-parametric method, following Harrison et al (2014)
    def __init__(self,wline,lineflux):

        cum=lineflux.cumsum()/lineflux.sum()
        cumfrac=np.array([0.02,0.05,0.10,0.25,0.5,0.75,0.90,0.95,0.98]) #percentiles
        (w02,w05,w10,w25,w50,w75,w90,w95,w98)=np.interp(cumfrac,cum,wline.value) 

        self.wave_02_98 = (w02,w98)
        self.wave_05_95 = (w05,w95)
        self.wave_10_90 = (w10,w90)
        self.wave_25_75 = (w25,w75)
        self.wave_50 = w50
        # compute the peak of the profile
        ## fit a parabola to the peak and find the maximum as the zero of the derivative
        pmax=lineflux.argmax()
        wline_par = wline[pmax-2:pmax+3].value
        lineflux_par = lineflux[pmax-2:pmax+3].value
        pol2 = np.polyfit(wline_par,lineflux_par,2)
        wave_peak = -pol2[1]/pol2[0]/2
        self.wpeak=wave_peak

    def W80(self): #measure of the line  width containing 80% of the flux
        w80 = (self.wave_10_90[1]-self.wave_10_90[0]) * speed / self.wpeak
        return w80  

    def dv(self): #measures the velocity offset of the broad wings of the emission line from systemic
        dv=((self.wave_05_95[0]+self.wave_05_95[1])/2.-self.wpeak) * speed / self.wpeak
        return dv 

    def vmed(self): #mean velocity
        vmed = (self.wave_50-self.wpeak) * speed / self.wpeak
        return vmed

    def asim(self): #it measures the assimetry of the line (according to Giovanna)		
        asim = (-1*(np.abs(self.wave_10_90[0]-self.wave_50)-np.abs(self.wave_10_90[1]-self.wave_50))) * speed / self.wpeak
        return asim

    def v05(self):
        v05 = (self.wave_05_95[0] - self.wpeak) * speed / self.wpeak
        return v05

    def v95(self):
        v95 = (self.wave_05_95[1] - self.wpeak)  * speed / self.wpeak
        return v95

    def v02(self):
        v02 = (self.wave_02_98[0] - self.wpeak) * speed / self.wpeak
        return v02

    def v98(self):
        v98 = (self.wave_02_98[1] - self.wpeak)  * speed / self.wpeak
        return v98

    def v10(self):
        v10 = (self.wave_10_90[0] - self.wpeak) * speed / self.wpeak
        return v10

    def v90(self):
        v90 = (self.wave_10_90[1] - self.wpeak) * speed / self.wpeak
        return v90

    def bestfit_nonpar(self):
        return np.array([self.wpeak,self.W80(),self.dv(),self.v05(),self.v95(),self.vmed(),self.asim()])
    
def perform_nonpar_plot(tgt_name,wave_reg,line_combo_fit,nonpar_fit,flux_norm):
    
    line_combo_fit_line = None
    
    if all('4959' in c.name for c in line_combo_fit)==True:
        for c in line_combo_fit:
            if '4959' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c
        
    else:
        for c in line_combo_fit:
            if '5007' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c
                
    wave_ext = np.linspace(min(wave_reg),max(wave_reg),1000)
    flux_ext = line_combo_fit_line(wave_ext)*flux_norm

    plt.figure(figsize=(9,7))
    xlabel=r"Velocity [km $s^{-1}$]"
    ylabel='$F_\lambda$ $[10^{-15}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    plt.plot(2.99e5*(wave_ext.value/nonpar_fit[0]-1),line_combo_fit_line(wave_ext)*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,linewidth=3)
    plt.axvline(0,linestyle='--',color='red',label=r'v$_p$')
    plt.axvline(nonpar_fit[3],linestyle='--',color='grey',label='v05')
    plt.axvline(nonpar_fit[4],linestyle='--',color='gray',label='v95')
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.legend(fontsize=14)
    plt.xlim(-2000,2000)
    plt.tick_params(axis='both', labelsize=12)
    plt.title(tgt_name,fontsize=18)
    plt.savefig('Graphs/NonParametric/'+tgt_name+'_nonpar.pdf',bbox_inches='tight')

    return None

################################################## GIOVANNA'S METHOD

def perform_outflow_plot(tgt_name,wave_reg,line_combo_fit,flux_norm):
    
    line_combo_fit_line = None

    if all('4959' in c.name for c in line_combo_fit)==True:
        for c in line_combo_fit:
            if '4959' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c

    else:
        for c in line_combo_fit:
            if '5007' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c

    wave_ext = np.linspace(min(wave_reg),max(wave_reg),1000)
    flux_ext = line_combo_fit_line(wave_ext)*flux_norm

    ppeak = flux_ext.argmax()

    wpeak = wave_ext[ppeak]
    flux_peak = flux_ext[ppeak]
    wave_vel = 2.99e5*(wave_ext/wpeak-1)

    pblue = (np.abs(flux_ext[np.where(wave_ext<=wpeak)]-np.max(flux_ext[np.where(wave_ext<=wpeak)])/3).argmin())
    pred = (np.abs(flux_ext[np.where(wave_ext>=wpeak)]-np.max(flux_ext[np.where(wave_ext>=wpeak)])/3).argmin())

    if len(wave_ext[:ppeak+1])>len(wave_ext[ppeak:]): #blue wing bigger than red wing
        wave_blue = wave_ext[:ppeak+1][-len(wave_ext[ppeak:]):]
        wave_blue_vel = wave_vel[:ppeak+1][-len(wave_ext[ppeak:]):]
        flux_blue = flux_ext[:ppeak+1][-len(wave_ext[ppeak:]):]
        wave_red = wave_blue
        wave_red_vel = -wave_blue_vel   
        flux_red = flux_ext[ppeak:][::-1]

    else:
        wave_blue = wave_ext[:ppeak+1]
        wave_blue_vel = wave_vel[:ppeak+1]
        flux_blue = flux_ext[:ppeak+1]
        wave_red = wave_blue
        wave_red_vel = -wave_blue_vel
        flux_red = flux_ext[ppeak:ppeak+len(wave_blue)][::-1]

    wave_outflow = wave_blue[np.where(wave_blue<wave_ext[pblue])]
    wave_outflow_vel = wave_blue_vel[np.where(wave_blue<wave_ext[pblue])]
    flux_outflow = (flux_blue-flux_red)[np.where(wave_blue<wave_ext[pblue])]
    flux_line_outflow = auc(wave_outflow,flux_outflow)

    cum = flux_outflow.cumsum()/flux_outflow.sum()
    v_outflow = np.interp(0.5,cum.value,wave_outflow_vel.value)

    plt.figure(figsize=(9,7))
    xlabel=r"Velocity [km $s^{-1}$]"
    ylabel='$F_\lambda$ $[10^{-15}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    plt.axvline(2.99e5*(wpeak/wpeak-1),color='black',linestyle='--',label=r'v$_p$')
    plt.plot(wave_vel,flux_ext*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,linewidth=3)

    if flux_line_outflow<0:
        if len(wave_ext[:ppeak+1])<len(wave_ext[ppeak:]): # red wing bigger than blue wing        
            wave_red = wave_ext[ppeak:][:len(wave_ext[:ppeak+1:])]
            wave_red_vel = wave_vel[ppeak:][:len(wave_ext[:ppeak+1:])]  
            flux_red = flux_ext[ppeak:][:len(wave_ext[:ppeak+1:])][::-1]
            wave_blue = wave_red
            wave_blue_vel = -wave_red_vel
            flux_blue = flux_ext[:ppeak+1][::-1]

        else:
            wave_red = wave_ext[ppeak:]
            wave_red_vel = wave_vel[ppeak:]  
            flux_red = flux_ext[ppeak:]
            wave_blue = wave_red
            wave_blue_vel = -wave_red_vel
            flux_blue = flux_ext[ppeak+1-len(wave_red):ppeak+1][::-1]

        wave_outflow = wave_red[np.where(wave_red>wave_ext[ppeak+pred])]
        wave_outflow_vel = wave_red_vel[np.where(wave_red>wave_ext[ppeak+pred])]
        flux_outflow = (flux_red-flux_blue)[np.where(wave_red>wave_ext[ppeak+pred])]
        flux_line_outflow = auc(wave_outflow,flux_outflow)

        cum = flux_outflow.cumsum()/flux_outflow.sum()
        v_outflow = np.interp(0.5,cum.value,wave_outflow_vel.value)

        plt.plot(wave_red_vel,flux_blue*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='black')
        plt.plot(wave_outflow_vel,flux_outflow*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='orange')
        plt.fill_between(wave_outflow_vel,flux_outflow*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='orange',label='Outflow')
        plt.axvspan(wave_vel[pblue].value,wave_vel[pred+ppeak].value,color='lightgray',label='Core')
        plt.xlim(-max(wave_red_vel),max(wave_red_vel))
        plt.tick_params(axis='both', labelsize=12)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        plt.legend(fontsize=14)    
        plt.title(tgt_name,fontsize=18)
        plt.savefig('Graphs/Outflow/SperanzaOutflow/'+tgt_name+'_outflow.pdf',bbox_inches='tight')

    else:    
        plt.plot(wave_blue_vel,flux_red*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='black')
        plt.plot(wave_outflow_vel,flux_outflow*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='orange')
        plt.fill_between(wave_outflow_vel,flux_outflow*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='orange',label='Outflow')
        plt.axvspan(wave_vel[pblue].value,wave_vel[pred+ppeak].value,color='lightgray',label='Core')
        plt.xlim(-max(wave_red_vel),max(wave_red_vel))
        plt.tick_params(axis='both', labelsize=12)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        plt.legend(fontsize=14)    
        plt.title(tgt_name,fontsize=18)
        plt.savefig('Graphs/Outflow/SperanzaOutflow/'+tgt_name+'_outflow.pdf',bbox_inches='tight') 
    
    return None

def outflow(wave_reg,line_combo_fit_line,flux_norm):

    wave_ext = np.linspace(min(wave_reg),max(wave_reg),1000)
    flux_ext = line_combo_fit_line(wave_ext)*flux_norm

    ppeak = flux_ext.argmax()

    wpeak = wave_ext[ppeak]
    flux_peak = flux_ext[ppeak]
    wave_vel = 2.99e5*(wave_ext/wpeak-1)

    pblue = (np.abs(flux_ext[np.where(wave_ext<=wpeak)]-np.max(flux_ext[np.where(wave_ext<=wpeak)])/3).argmin())
    pred = (np.abs(flux_ext[np.where(wave_ext>=wpeak)]-np.max(flux_ext[np.where(wave_ext>=wpeak)])/3).argmin())

    if len(wave_ext[:ppeak+1])>len(wave_ext[ppeak:]): #blue wing bigger than red wing
        wave_blue = wave_ext[:ppeak+1][-len(wave_ext[ppeak:]):]
        wave_blue_vel = wave_vel[:ppeak+1][-len(wave_ext[ppeak:]):]
        flux_blue = flux_ext[:ppeak+1][-len(wave_ext[ppeak:]):]
        wave_red = wave_blue
        wave_red_vel = -wave_blue_vel   
        flux_red = flux_ext[ppeak:][::-1]

    else:
        wave_blue = wave_ext[:ppeak+1]
        wave_blue_vel = wave_vel[:ppeak+1]
        flux_blue = flux_ext[:ppeak+1]
        wave_red = wave_blue
        wave_red_vel = -wave_blue_vel
        flux_red = flux_ext[ppeak:ppeak+len(wave_blue)][::-1]

    wave_outflow = wave_blue[np.where(wave_blue<wave_ext[pblue])]
    wave_outflow_vel = wave_blue_vel[np.where(wave_blue<wave_ext[pblue])]
    flux_outflow = (flux_blue-flux_red)[np.where(wave_blue<wave_ext[pblue])]
    flux_line_outflow = auc(wave_outflow,flux_outflow)

    cum = flux_outflow.cumsum()/flux_outflow.sum()
    v_outflow = np.interp(0.5,cum.value,wave_outflow_vel.value)

    if flux_line_outflow<0:
        if len(wave_ext[:ppeak+1])<len(wave_ext[ppeak:]): # red wing bigger than blue wing        
            wave_red = wave_ext[ppeak:][:len(wave_ext[:ppeak+1:])]
            wave_red_vel = wave_vel[ppeak:][:len(wave_ext[:ppeak+1:])]  
            flux_red = flux_ext[ppeak:][:len(wave_ext[:ppeak+1:])][::-1]
            wave_blue = wave_red
            wave_blue_vel = -wave_red_vel
            flux_blue = flux_ext[:ppeak+1][::-1]

        else:
            wave_red = wave_ext[ppeak:]
            wave_red_vel = wave_vel[ppeak:]  
            flux_red = flux_ext[ppeak:]
            wave_blue = wave_red
            wave_blue_vel = -wave_red_vel
            flux_blue = flux_ext[ppeak+1-len(wave_red):ppeak+1][::-1]

        wave_outflow = wave_red[np.where(wave_red>wave_ext[ppeak+pred])]
        wave_outflow_vel = wave_red_vel[np.where(wave_red>wave_ext[ppeak+pred])]
        flux_outflow = (flux_red-flux_blue)[np.where(wave_red>wave_ext[ppeak+pred])]
        flux_line_outflow = auc(wave_outflow,flux_outflow)

        cum = flux_outflow.cumsum()/flux_outflow.sum()
        v_outflow = np.interp(0.5,cum.value,wave_outflow_vel.value)
    
    return np.array([flux_line_outflow,v_outflow])

def outflow_energetics(outflow_L,outflow_v):
    
    n_e = 200 #cm^-3
    L_s = 3.846e33 #erg/s
    M_s = 1.98847e30 #kg
    
    outflow_radio = 1*3.086e19 #m

    outflow_mass = (4e7)*(1/10**0.01)*(outflow_L/1e44)*(1e3/n_e)*M_s #in kg
    outflow_kin_E = 0.5*outflow_mass*outflow_v**2 #in J = kg*m^2/s^2

    outflow_mass_rate = 3*np.abs(outflow_v)*(outflow_mass/outflow_radio) #in kg/s       
    
    outflow_E_rate = 0.5*outflow_mass_rate*outflow_v**2 #in J/s

    outflow_energy = np.array([np.log10(outflow_mass/M_s),np.log10(outflow_kin_E*10**7),outflow_radio/3.086e19,np.log10(outflow_E_rate*10**2),outflow_mass_rate*3.154e7/M_s],dtype='float')
    
    return outflow_energy

def outflow_energetics_uncertainties(m,Nsimul):
    
    outflow_energy_0 = outflow_energetics(m['Loutflow'],m['Voutflow'])
    params_simul = np.zeros((len(outflow_energy_0),Nsimul))
    outflow_simul = np.zeros((2,Nsimul))

    for isimul in range(Nsimul):

        L_sim = rnd.normal(m['Loutflow'],m['eLoutflow'])
        v_sim = rnd.normal(m['Voutflow']*10**3,m['eVoutflow']*10**3)
        
        outflow_simul[:,isimul] = np.array([np.float(L_sim),np.float(v_sim)])

        params_simul[:,isimul] = outflow_energetics(L_sim,v_sim)

    outflow_energy = np.nanmedian(params_simul,axis=1)
    outflow_energy_errors = np.nanstd(params_simul,axis=1)

    return outflow_energy,outflow_energy_errors

############################################# NONPAR METHOD

###### V05 AND V95

def outflow_nonpar(wave_reg,line_combo_fit_line,nonpar_fit,flux_norm):

    wave_ext = np.linspace(min(wave_reg),max(wave_reg),1000)
    flux_ext = line_combo_fit_line(wave_ext)*flux_norm
    
    ppeak = flux_ext.argmax()
    wpeak = wave_ext[ppeak]
    wave_vel = 2.99e5*((wave_ext)/wpeak-1)

    v10 = nonpar_fit[3]
    v90 = nonpar_fit[4]

    p10 = np.abs(np.abs(wave_vel[np.where(wave_vel<=0)])-np.abs(v10)).argmin()
    p90 = np.abs(np.abs(wave_vel[np.where(wave_vel>=0)])-np.abs(v90)).argmin()

    wave10 = wave_ext[np.where(wave_vel<=0)][:p10]/u.Angstrom
    wave90 = wave_ext[np.where(wave_vel>=0)][p90:]/u.Angstrom
    wave_vel_10 = wave_vel[np.where(wave_vel<=0)][:p10]
    wave_vel_90 = wave_vel[np.where(wave_vel>=0)][p90:]
    flux10 = flux_ext[np.where(wave_vel<=0)][:p10]
    flux90 = flux_ext[np.where(wave_vel>=0)][p90:]

    flux_line_blue = auc(wave10,flux10)
    cum_blue = flux10.cumsum()/flux10.sum()
    v_blue = np.interp(0.5,cum_blue,wave_vel_10)

    flux_line_red = auc(wave90,flux90)
    cum_red = flux90.cumsum()/flux90.sum()
    v_red = np.interp(0.5,cum_red,wave_vel_90)
    
    return np.array([flux_line_blue,v_blue]),np.array([flux_line_red,v_red])

def perform_outflow_nonpar_plot(tgt_name,wave_reg,line_combo_fit,nonpar_fit,flux_norm):
    
    line_combo_fit_line = None
    
    if all('4959' in c.name for c in line_combo_fit)==True:
        for c in line_combo_fit:
            if '4959' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c
        
    else:
        for c in line_combo_fit:
            if '5007' in c.name:
                if isinstance(line_combo_fit_line,Model):
                    line_combo_fit_line = line_combo_fit_line + c
                else:
                    line_combo_fit_line = c
                
    wave_ext = np.linspace(min(wave_reg),max(wave_reg),1000)
    flux_ext = line_combo_fit_line(wave_ext)*flux_norm
    
    ppeak = flux_ext.argmax()
    wpeak = wave_ext[ppeak]
    wave_vel = 2.99e5*((wave_ext)/wpeak-1)

    v10 = nonpar_fit[3]
    v90 = nonpar_fit[4]

    p10 = np.abs(np.abs(wave_vel[np.where(wave_vel<=0)])-np.abs(v10)).argmin()
    p90 = np.abs(np.abs(wave_vel[np.where(wave_vel>=0)])-np.abs(v90)).argmin()

    wave10 = wave_ext[np.where(wave_vel<=0)][:p10]/u.Angstrom
    wave90 = wave_ext[np.where(wave_vel>=0)][p90:]/u.Angstrom
    wave_vel_10 = wave_vel[np.where(wave_vel<=0)][:p10]
    wave_vel_90 = wave_vel[np.where(wave_vel>=0)][p90:]
    flux10 = flux_ext[np.where(wave_vel<=0)][:p10]
    flux90 = flux_ext[np.where(wave_vel>=0)][p90:]
    plt.figure(figsize=(9,7))
    xlabel=r"Velocity [km $s^{-1}$]"
    ylabel='$F_\lambda$ $[10^{-15}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$'
    plt.axvline(2.99e5*(wpeak/wpeak-1),color='black',linestyle='--',label=r'v$_p$')
    plt.plot(wave_vel,flux_ext*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,linewidth=3)
    plt.axvspan(v10,v90,color='lightgray',label=r'v$_{05}$-v$_{95}$')
    plt.fill_between(wave_vel_10,flux10*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='lightblue',label='Blue outflow')
    plt.fill_between(wave_vel_90,flux90*(u.A*u.cm*u.cm*u.s/u.erg)/1.0e-15,color='lightcoral',label='Red outflow')
    plt.xlim(-2500,2500)
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.legend(fontsize=14)    
    plt.title(tgt_name,fontsize=18)
    plt.savefig('Graphs/Outflow/NonParOutflow/'+tgt_name+'_outflow.pdf',bbox_inches='tight')
    
    return None

#### ENERGETICS

def outflow_energetics_nonpar(outflow_blue_L,outflow_blue_V,outflow_red_L,outflow_red_V):
    
    n_e = 200 #cm^-3
    L_s = 3.846e33 #erg/s
    M_s = 1.98847e30 #kg
    
    outflow_radio = 1*3.086e19 #m
    
    outflow_mass_blue = (4e7)*(1/10**0.01)*(outflow_blue_L/1e44)*(1e3/n_e)*M_s #in kg
    outflow_mass_red = (4e7)*(1/10**0.01)*(outflow_red_L/1e44)*(1e3/n_e)*M_s #in kg
    outflow_mass = outflow_mass_blue + outflow_mass_red
    outflow_kin_E = 0.5*outflow_mass_blue*outflow_blue_V**2 + 0.5*outflow_mass_red*outflow_red_V**2#in J = kg*m^2/s^2

    outflow_mass_rate_blue = 3*np.abs(outflow_blue_V)*(outflow_mass_blue/outflow_radio)
    outflow_mass_rate_red = 3*np.abs(outflow_red_V)*(outflow_mass_red/outflow_radio) #in kg/s     
    outflow_mass_rate = outflow_mass_rate_blue + outflow_mass_rate_red
    outflow_E_rate = 0.5*outflow_mass_rate_blue*outflow_blue_V**2 + 0.5*outflow_mass_rate_red*outflow_red_V**2 #in J/s

    outflow_energy = np.array([np.log10(outflow_mass/M_s),np.log10(outflow_kin_E*10**7),outflow_radio/3.086e19,np.log10(outflow_E_rate*10**2),outflow_mass_rate*3.154e7/M_s],dtype='float')
    
    return outflow_energy


def outflow_energetics_nonpar_uncertainties(z,Nsimul):
    
    outflow_energy_0 = outflow_energetics_nonpar(z['Loutflow'][0],z['Voutflow'][0],z['Loutflow'][1],z['Voutflow'][1])
    params_simul = np.zeros((len(outflow_energy_0),Nsimul))

    for isimul in range(Nsimul):
        
        outflow_blue_L = rnd.normal(z['Loutflow'][0],z['eLoutflow'][0])
        outflow_blue_V = rnd.normal(z['Voutflow'][0]*10**3,z['eVoutflow'][0]*10**3)
        outflow_red_L = rnd.normal(z['Loutflow'][1],z['eLoutflow'][1])
        outflow_red_V = rnd.normal(z['Voutflow'][1]*10**3,z['eVoutflow'][1]*10**3)
        
        params_simul[:,isimul] = outflow_energetics_nonpar(outflow_blue_L,np.abs(outflow_blue_V),outflow_red_L,np.abs(outflow_red_V))
        
    outflow_energy = np.nanmedian(params_simul,axis=1)
    outflow_energy_errors = np.nanstd(params_simul,axis=1)

    return outflow_energy,outflow_energy_errors


