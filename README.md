# SkinCare ML

Componente de aprendizaje automático del proyecto
[SkinCare](https://github.com/sogekingdabest/SkinCareApp). Incluye código para
preparar datos, entrenar modelos y exportarlos a TensorFlow Lite.

Es un proyecto educativo y experimental. No ofrece diagnósticos médicos ni
sustituye la valoración de un profesional sanitario.

## Código

- [Notebook principal](notebooks/skincare_training.ipynb).
- [Notebook de validación de captura](notebooks/capture_validity_training.ipynb).
- [Preparación y evaluación](scripts/).
- [Kernels de Kaggle](kaggle/).

Los notebooks están preparados para Kaggle. Los datasets se adjuntan mediante
los metadatos de cada kernel; las imágenes y los modelos generados se conservan
fuera de Git.

## Verificación local

Con Python 3.12, desde la raíz del repositorio en PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force .test-tmp | Out-Null
$env:TMP = (Resolve-Path .test-tmp).Path
$env:TEMP = $env:TMP
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scripts/validate_notebook.py
.venv/Scripts/python.exe scripts/validate_capture_validity_notebook.py
.venv/Scripts/python.exe scripts/validate_export_kernel.py
```

Estas comprobaciones no entrenan los modelos ni requieren acceso a Kaggle.

## Licencia

Código bajo [GPL-3.0](LICENSE). Los datasets tienen sus propias condiciones de uso.
