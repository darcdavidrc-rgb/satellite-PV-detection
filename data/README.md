# Datos
```
import os
!pip install roboflow -q

from roboflow import Roboflow

rf = Roboflow(api_key="") #Solicitar por correo en david.romero@estudiantes.cicy.mx
project = rf.workspace("daves-workspace-cvhyt").project("satellite-pv")
version = project.version(2)
dataset = version.download("yolov8")
print(f"✓ Dataset descargado en: {dataset.location}")

