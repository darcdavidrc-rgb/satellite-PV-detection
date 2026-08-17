# satellite-PV-detection
Repository for training and evaluating a model to detect photovoltaic systems using satellite imagery.

Benchmark comparativo entre modelos basados en atención (*Dynamic Window Vision Transformer - DW-ViT*) y modelos convolucionales de segmentación de instancias (*YOLOv9/v8 Segmentation*) para la detección, delimitación poligonal y estimación de módulos solares en imágenes satelitales.

## Resultados Clave

| Métrica | DW-ViT (From Scratch) | YOLO-Seg (Transfer Learning) |
| :--- | :---: | :---: |
| **MAE (Conteo)** | 19.25 paneles | **8.31 paneles** |
| **RMSE** | 31.81 | **15.66** |
| **$R^2$** | -0.5255 | **0.6302** |
| **mAP@50 (Box / Mask)** | N/A | **64.59% / 46.21%** |
| **mIoU (Máscara Poligonal)**| N/A | **42.64%** |

## Estructura del Repositorio
- `notebooks/`: Flujos de entrenamiento y evaluación reproducibles en Google Colab.
- `src/`: Definición de arquitecturas y scripts de entrenamiento independientes.
- `results/`: Gráficas de dispersión, residuales y distribución de IoU.

## Replicación
```bash
git clone [https://github.com/darcdavidrc-rgb/satellite-pv-detection.git](https://github.com/darcdavidrc-rgb/satellite-pv-detection.git)
cd satellite-pv-detection
pip install -r requirements.txt
```
