# CNN simple para clasificacion de imagenes

## URL principal de entrega

Formulario de submission:

```text
https://github.com/alonso59/nayarit_challenge/issues/new?template=model_submission.yml
```

Debes subir un archivo `submission.zip` con `predictions.csv` y `ABLATIONS.md`.

Este proyecto entrena una CNN sencilla con PyTorch. El flujo esta separado en tres pasos:

- entrenamiento con imagenes etiquetadas,
- evaluacion local en `val` con etiquetas,
- prediccion del `test` sin etiquetas para generar `predictions.csv`.

Los comandos de este README asumen que estas en la raiz del proyecto, es decir, en la carpeta que contiene `src/`, `data/` y `dataset/`.

## Dataset del reto

Descarga directa:

[stl5_challenge_public.zip](https://github.com/alonso59/nayarit_base_cnn/raw/main/stl5_challenge_public.zip)

Linux/macOS:

```bash
curl -L -o stl5_challenge_public.zip https://github.com/alonso59/nayarit_base_cnn/raw/main/stl5_challenge_public.zip
unzip -q stl5_challenge_public.zip
rm -rf dataset
mkdir dataset
cp -R stl5_challenge/public/. dataset/
```

Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/alonso59/nayarit_base_cnn/raw/main/stl5_challenge_public.zip" -OutFile "stl5_challenge_public.zip"
Expand-Archive -Path "stl5_challenge_public.zip" -DestinationPath "." -Force
if (Test-Path dataset) { Remove-Item dataset -Recurse -Force }
New-Item -ItemType Directory -Path dataset | Out-Null
Copy-Item -Path "stl5_challenge/public/*" -Destination "dataset" -Recurse
```

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Archivos principales

Normalmente solo debes modificar:

- `src/config.yaml`: rutas, dataset, folds e hiperparametros.
- `src/augmentations.py`: transformaciones de entrenamiento y validacion.
- `src/model.py`: arquitectura de la CNN.

La infraestructura esta en:

- `src/dataloader.py`: carga de datos etiquetados y no etiquetados.
- `src/trainer.py`: loop de entrenamiento.
- `src/train.py`: entrenamiento.
- `src/eval.py`: evaluacion local con etiquetas.
- `src/predict.py`: prediccion sin etiquetas para leaderboard.

## 3. Eval vs predict

Usa `src/eval.py` para evaluar el conjunto de validacion etiquetado:

- lee `dataset/val/`,
- requiere carpetas de clase,
- calcula metricas como loss, accuracy, F1 y matriz de confusion.

Usa `src/predict.py` para generar el archivo de entrega del challenge:

- lee `dataset/test/images/`,
- no requiere etiquetas ni carpetas de clase,
- no calcula metricas,
- genera `predictions.csv`.

Para el leaderboard, los estudiantes deben entregar `predictions.csv` y
`ABLATIONS.md` dentro de `submission.zip` en el formulario de submission:

```text
https://github.com/alonso59/nayarit_challenge/issues/new?template=model_submission.yml
```

## 4. Estructura del dataset para el challenge

El dataset esperado para estudiantes es:

```text
dataset/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── test/
    └── images/
        ├── test_000001.png
        ├── test_000002.png
        └── test_000003.png
```

Reglas importantes:

- `train` tiene etiquetas y se usa para entrenar.
- `val` tiene etiquetas y se usa para calcular metricas locales.
- `test/images` no tiene etiquetas y se usa solo para generar `predictions.csv`.
- `test` no debe tener carpetas de clases.
- Las carpetas de clases etiquetadas deben llamarse `0`, `1`, `2`, etc.

## 5. Configuracion

El config principal esta en `src/config.yaml`. Los scripts aceptan `--config src/config.yaml` para indicar explicitamente que archivo de configuracion usar.

Ejemplo para el challenge:

```yaml
seed: 42
device: auto
output_dir: output/${now:%Y%m%d_%H%M%S}

dataset:
  name: folder
  root: ./data
  path: ./dataset
  class_names: ["airplane", "bird", "car", "cat", "dog"]
  download: true
  image_size: 96
  batch_size: 64
  num_workers: 2
  num_folds: 5
  fold_index: 0

training:
  epochs: 5
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  betas: [0.9, 0.999]
```

Campos comunes:

- `dataset.path`: carpeta del dataset del challenge, normalmente `./dataset`.
- `dataset.image_size`: tamano al que se redimensionan las imagenes.
- `dataset.batch_size`: imagenes por lote.
- `training.epochs`: numero de epocas.
- `training.optimizer`: `sgd` o `adam`.
- `training.learning_rate`: tasa de aprendizaje.

## 6. Entrenar

```bash
python src/train.py --config src/config.yaml
```

Si no quieres usar terminal o estas trabajando en Google Colab, abre `run_train.ipynb`. La notebook instala dependencias, ejecuta entrenamiento, evalua validacion y genera `predictions.csv`.

Tambien puedes seguir usando Hydra directamente:

```bash
python src/train.py training.epochs=10 training.learning_rate=0.001
```

Cada entrenamiento guarda:

```text
output/YYYYMMDD_HHMMSS/
  .hydra/config.yaml
  model.pt
  plots/training.png
  checkpoints/best.pt
  checkpoints/last.pt
```

## 7. Evaluar validacion etiquetada

`src/eval.py` evalua solo el split `val`. No usa `dataset/test/images` y no asume etiquetas ocultas.

```bash
python src/eval.py --config src/config.yaml --checkpoint output/YYYYMMDD_HHMMSS/checkpoints/best.pt
```

La evaluacion calcula metricas locales como:

- `loss`
- `accuracy`
- `precision_macro`
- `sensitivity_macro`
- `specificity_macro`
- `f1_macro`
- matriz de confusion

Los archivos se guardan en:

```text
output/YYYYMMDD_HHMMSS/eval/
  confusion_matrix.png
  roc_curve.png
  metrics.txt
```

## 8. Generar predicciones para leaderboard

`src/predict.py` carga imagenes desde:

```text
dataset/test/images/
```

No espera carpetas de clases y no calcula metricas. Genera un CSV con el formato exacto:

```csv
id,y_pred
test_000001,0
test_000002,3
test_000003,4
```

Comando:

```bash
python src/predict.py --config src/config.yaml --checkpoint output/YYYYMMDD_HHMMSS/checkpoints/best.pt --output predictions.csv
```

El `id` sale del nombre del archivo sin extension. Por ejemplo:

```text
dataset/test/images/test_000001.png -> test_000001
```

La prediccion usa la clase con mayor logit:

```python
y_pred = logits.argmax(dim=1)
```

## 9. Antes de entregar

Revisa:

- `dataset/train` y `dataset/val` tienen carpetas numericas de clases.
- `dataset/test/images` contiene imagenes directamente, sin carpetas de clase.
- `src/config.yaml` tiene `dataset.name: folder`.
- `dataset.image_size` coincide con el tamano que quieres alimentar a la CNN.
- `predictions.csv` tiene las columnas exactas `id,y_pred`.
- Escribiste `ABLATIONS.md` con tus experimentos o ablations.
- Creaste `submission.zip` con `predictions.csv` y `ABLATIONS.md` en la raiz del ZIP.
- El formulario de entrega esta aqui:
  `https://github.com/alonso59/nayarit_challenge/issues/new?template=model_submission.yml`

Linux/macOS:

```bash
zip submission.zip predictions.csv ABLATIONS.md
```

Windows PowerShell:

```powershell
Compress-Archive -Path predictions.csv, ABLATIONS.md -DestinationPath submission.zip -Force
```
