# Paalpha  18756.1 (vacuum), HeIIλ1.8637μm, HeIλ1.8691
## continuum region 20770:20820, 21400:21450
## peaks 21032, 21091, 21158

wave_OIIIb = 4958.911
wave_OIIIr = 5006.843

wave_NIIb = 6548.05
wave_NIIr = 6583.45
wave_Ha=6562.79

def tie_stddevn_OIIIn(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007n'].stddev.value * velratio
     return stddev
     
def tie_stddevn_OIIIn1(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007n1'].stddev.value * velratio
     return stddev
 
def tie_stddevn_OIIIn2(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007n2'].stddev.value * velratio
     return stddev
         
def tie_stddevn_OIIIi(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007i'].stddev.value * velratio
     return stddev
     
def tie_stddevn_OIIIi1(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007i1'].stddev.value * velratio
     return stddev
     
def tie_stddevn_OIIIi2(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007i2'].stddev.value * velratio
     return stddev
     
def tie_stddevn_OIIIb(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007b'].stddev.value * velratio
     return stddev
     
def tie_stddevn_OIIIr(model):
     velratio = wave_OIIIb/wave_OIIIr
     stddev = model['[OIII]5007r'].stddev.value * velratio
     return stddev
     
     
          
     
def tie_center_OIIIn(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007n'].mean.value * velratio
     return center
     
def tie_center_OIIIn1(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007n1'].mean.value * velratio
     return center
 
def tie_center_OIIIn2(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007n2'].mean.value * velratio
     return center
         
def tie_center_OIIIi(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007i'].mean.value * velratio
     return center
         
def tie_center_OIIIi1(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007i1'].mean.value * velratio
     return center
           
def tie_center_OIIIi2(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007i2'].mean.value * velratio
     return center
     
def tie_center_OIIIb(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007b'].mean.value * velratio
     return center 
        
def tie_center_OIIIr(model):
     velratio = wave_OIIIb/wave_OIIIr
     center = model['[OIII]5007r'].mean.value * velratio
     return center
     
   
def tie_ampl_OIIIn(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007n'].amplitude.value * OIIIb_r
     return amp

def tie_ampl_OIIIn1(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007n1'].amplitude.value * OIIIb_r
     return amp

def tie_ampl_OIIIn2(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007n2'].amplitude.value * OIIIb_r
     return amp
     
def tie_ampl_OIIIi(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007i'].amplitude.value * OIIIb_r
     return amp
     
def tie_ampl_OIIIi1(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007i1'].amplitude.value * OIIIb_r
     return amp
     
def tie_ampl_OIIIi2(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007i2'].amplitude.value * OIIIb_r
     return amp
     
def tie_ampl_OIIIb(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007b'].amplitude.value * OIIIb_r
     return amp
     
def tie_ampl_OIIIr(model):
     OIIIb_r = 1./3.
     amp = model['[OIII]5007r'].amplitude.value * OIIIb_r
     return amp
    
def tie_center_OH(model):
     velratio = wave_H/wave_O
     center = model['[OIII]5007'].mean.value * velratio
     return center
