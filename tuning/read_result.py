import rpy2.robjects as ro
import os
from datetime import datetime

r = ro.r

rdata_file = r"C:\Users\CristinaTobarFernánd\Desktop\DOCTORADO\Artículos_Realizados\3-TOPsimlearheuristic\TeamOriented\tuning\irace.log"
r["load"](rdata_file)

iraceResults = r["iraceResults"]  # <- aquí “entras”
print(type(iraceResults))
state = iraceResults.rx2("state")
print(state.names)


mtime = os.path.getmtime(rdata_file)
fecha_mod = datetime.fromtimestamp(mtime)

print(r("names(iraceResults$state)"))

mtime = os.path.getmtime(rdata_file)
fecha_mod = datetime.fromtimestamp(mtime)

print("Última modificación:", fecha_mod)

print(r("iraceResults$state$experiment_log"))
