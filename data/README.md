# Datos

```python
!pip install roboflow -q

import os
from roboflow import Roboflow

# Solicitar API Key por correo: david.romero@estudiantes.cicy.mx
rf = Roboflow(api_key="TU_API_KEY") 
project = rf.workspace("daves-workspace-cvhyt").project("satellite-pv")
version = project.version(2)
dataset = version.download("yolov8")

print(f"✓ Dataset descargado en: {dataset.location}")
```
