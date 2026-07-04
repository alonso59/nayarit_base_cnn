# Reto de tabla de posiciones con CNN

## 1. Objetivo

El objetivo de este reto es construir una red neuronal convolucional (CNN) que clasifique imagenes de un dataset personalizado de 5 clases.

Entrenaras tu modelo usando las particiones publicas `train` y `val`. La tabla de posiciones final se calculara usando una particion `test` con etiquetas ocultas. No recibiras las etiquetas de test. En su lugar, debes generar un archivo `predictions.csv` y enviarlo al repositorio de la tabla de posiciones.

El reto no consiste solamente en obtener la mayor exactitud de validacion. Tu modelo debe generalizar a imagenes de test no vistas usando una arquitectura CNN clara, aumentos de datos apropiados y entrenamiento reproducible.

## 2. Que incluye este repositorio

Este repositorio ya contiene un pipeline basico de entrenamiento con PyTorch:

```text
src/
├── config.yaml         # configuracion del experimento
├── dataloader.py       # carga de train/val etiquetados y test sin etiquetas
├── augmentations.py    # transformaciones de entrenamiento y validacion
├── model.py            # arquitectura CNN por completar
├── trainer.py          # loop de entrenamiento
├── train.py            # punto de entrada para entrenar
├── eval.py             # evaluacion local en validacion
└── predict.py          # prediccion de test y generacion del CSV
```

El flujo esperado es:

```text
1. Completar el modelo CNN en src/model.py.
2. Elegir aumentos de datos razonables en src/augmentations.py.
3. Configurar el experimento en src/config.yaml.
4. Entrenar el modelo con src/train.py.
5. Evaluar localmente en val con src/eval.py.
6. Generar predictions.csv para el test sin etiquetas con src/predict.py.
7. Enviar predictions.csv al repositorio de la tabla de posiciones.
```

## 3. Descarga del dataset

Descarga el dataset del reto desde:

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

Despues de descargarlo y extraerlo, coloca el dataset en el repositorio con esta estructura:

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
        └── ...
```

Importante:

- `train` tiene etiquetas mediante carpetas de clase.
- `val` tiene etiquetas mediante carpetas de clase.
- `test/images` no tiene etiquetas.
- `test` no debe contener carpetas de clase.
- Las etiquetas de test estan ocultas y solo las conserva el instructor.

## 4. Clases

El dataset contiene 5 clases:

| Indice de clase | Nombre de clase |
|---:|---|
| 0 | [TODO: class_0] |
| 1 | [TODO: class_1] |
| 2 | [TODO: class_2] |
| 3 | [TODO: class_3] |
| 4 | [TODO: class_4] |

Tu archivo `predictions.csv` debe usar el indice numerico de la clase, no el nombre de la clase.

## 5. Instalacion

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Si usas Google Colab, puedes usar la notebook incluida:

```text
run_train.ipynb
```

## 6. Configuracion

Edita:

```text
src/config.yaml
```

Para el reto, los campos importantes son:

```yaml
dataset:
  name: folder
  path: ./dataset
  image_size: 200 # tamanio de input de la imagen que depende directamente de tu modelo propuesto y el numero de parametros
  batch_size: 128 # elegir este valor dependiendo de la cantidad de memoria disponible
  num_workers: 2 # batch_size / 4

training:
  epochs: 1
  optimizer: adam o sgd
  learning_rate: 0.001 o 1e-3 o 1e-2 o 1e-4
  weight_decay: 0.0 o 0.0001
  betas: [0.9, 0.999] generalmente fijos
```

Puedes cambiar estos valores durante la experimentacion.

## 7. Tarea 1: completar el modelo CNN

Abre:

```text
src/model.py
```

Debes implementar la clase `SimpleCNN`.

El modelo debe:

- recibir un tensor de entrada con forma `[batch_size, channels, height, width]`,
- usar al menos una capa `Conv2d`,
- producir logits con forma `[batch_size, num_classes]`,
- no aplicar `softmax` dentro de `forward`, porque `CrossEntropyLoss` espera logits crudos.

Una CNN valida puede usar componentes como:

```text
Conv2d
ReLU
MaxPool2d
BatchNorm2d
Dropout
Flatten
Linear
AdaptiveAvgPool2d
```

No uses modelos preentrenados.

## 8. Tarea 2: elegir aumentos de datos

Abre:

```text
src/augmentations.py
```

Puedes agregar transformaciones aleatorias solo dentro de `train_augmentations`.

Ejemplos de aumentos permitidos durante entrenamiento:

```text
RandomHorizontalFlip
RandomRotation
RandomAffine
ColorJitter
RandomCrop
RandomResizedCrop
GaussianBlur mediante RandomApply
```

No agregues transformaciones aleatorias al preprocesamiento de validacion o test. Validacion y test deben usar preprocesamiento determinista.

## 9. Entrenamiento

Ejecuta:

```bash
python src/train.py --config src/config.yaml
```

El script de entrenamiento guarda los resultados en una carpeta con marca de tiempo:

```text
output/YYYYMMDD_HHMMSS/
├── model.pt
├── plots/training.png
└── checkpoints/
    ├── best.pt
    └── last.pt
```

Usa `checkpoints/best.pt` para validacion y prediccion, a menos que tengas una razon justificada para usar otro checkpoint.

## 10. Evaluacion local en validacion

Usa `eval.py` para evaluar la particion de validacion etiquetada:

```bash
python src/eval.py --config src/config.yaml --checkpoint output/YYYYMMDD_HHMMSS/checkpoints/best.pt
```

Esto calcula metricas locales en `dataset/val/`, incluyendo:

```text
loss
accuracy
precision_macro
sensitivity_macro
specificity_macro
f1_macro
auroc_macro
confusion_matrix.png
roc_curve.png
```

Estas metricas de validacion son solo para desarrollo local. No son la puntuacion final de la tabla de posiciones.

## 11. Generar predicciones para la tabla de posiciones

Usa `predict.py` para generar predicciones para la particion de test sin etiquetas:

```bash
python src/predict.py \
  --config src/config.yaml \
  --checkpoint output/YYYYMMDD_HHMMSS/checkpoints/best.pt \
  --output predictions.csv
```

El archivo de salida debe llamarse:

```text
predictions.csv
```

El formato requerido es:

```csv
id,y_pred
test_000001,0
test_000002,3
test_000003,4
```

Reglas:

- El `id` debe coincidir con el nombre del archivo de imagen de test sin extension.
- El valor `y_pred` debe ser un indice entero de clase entre `0` y `4`.
- El CSV debe contener exactamente una prediccion por cada imagen de test.
- No modifiques los nombres de archivo de las imagenes de test.
- No incluyas probabilidades en el CSV enviado.

## 12. Repositorio de entrega

Envia tu `predictions.csv` aqui:

```text
[TODO: SUBMISSION_REPOSITORY_URL]
```

Metodo de entrega:

1. Abre el repositorio de entrega.
2. Crea un issue nuevo usando el formulario `Model submission`.
3. Completa la informacion de participante.
4. Proporciona un enlace a tu archivo `predictions.csv`.
5. Envia el issue.

Puedes enviar multiples veces. La tabla de posiciones publica conservara solo la ultima entrega valida de cada participante.

## 13. Multiples entregas

Cada participante puede enviar mas de una vez.

Regla de la tabla de posiciones:

```text
Solo se muestra en la tabla publica la ultima entrega valida de cada participante.
```

Una entrega es valida solo si:

- el CSV tiene las columnas correctas: `id,y_pred`,
- todos los IDs de test estan presentes,
- no hay IDs duplicados,
- todas las etiquetas predichas son enteros validos de `0` a `4`,
- el participante usa el mismo `team_id` en todas sus entregas.

## 14. Restricciones

Permitido:

```text
CNNs entrenadas desde cero
BatchNorm
Dropout
Aumento de datos
Ajuste de learning rate
SGD o Adam
Weight decay
Bloques pequenos de estilo residual implementados por el participante
```

No permitido:

```text
Modelos preentrenados
Vision Transformers
Datasets externos
Etiquetado manual del conjunto de test
Entrenar con imagenes de test
Usar etiquetas ocultas
Herramientas AutoML
Enviar predicciones generadas por otro participante
```

## 15. Reporte final sugerido

Cada participante debe preparar un resumen tecnico corto que incluya:

```text
1. Arquitectura del modelo
2. Numero de parametros entrenables
3. Configuracion de entrenamiento
4. Aumentos de datos usados
5. Metricas de validacion
6. Matriz de confusion en validacion
7. Al menos un ejemplo fallido o analisis de errores
8. Principal decision de diseno que mejoro el modelo
```

Se recomienda ampliamente incluir una tabla de ablacion:

| Experimento | Arquitectura / cambio | Params | Val accuracy | Val F1 macro | Comentario |
|---|---|---:|---:|---:|---|
| Baseline | CNN simple | [TODO] | [TODO] | [TODO] | [TODO] |
| + Augmentation1 | se agregaron transformaciones aleatorias | [TODO] | [TODO] | [TODO] | [TODO] |
| + Augmentation2 | se agregaron transformaciones aleatorias | [TODO] | [TODO] | [TODO] | [TODO] |
| + Augmentation3 | se agregaron transformaciones aleatorias | [TODO] | [TODO] | [TODO] | [TODO] |

## 16. Politica de evaluacion

La puntuacion de la tabla de posiciones se calculara usando etiquetas ocultas de test.

Puntuacion de la leaderboard:

```text
final_score = 0.70 * macro_f1 + 0.20 * accuracy + 0.10 * efficiency_score
```

```text
efficiency_score: numero de parametros del modelo
```
Si se usa una puntuacion de eficiencia, el instructor publicara la regla de conteo de parametros antes de la primera entrega oficial.

## 17. Fechas limite

| Hito | Fecha |
|---|---|
| Dataset publicado | [TODO: DATE] |
| Primera entrega valida | [TODO: DATE] |
| Fecha limite de entrega final | [TODO: DATE] |
| Presentacion final | [TODO: DATE] |

## 18. Checklist final antes de entregar

Antes de enviar, verifica:

```text
[ ] src/model.py tiene una CNN funcional.
[ ] El entrenamiento termina sin errores.
[ ] eval.py corre en val.
[ ] predict.py crea predictions.csv.
[ ] predictions.csv tiene las columnas id,y_pred.
[ ] Los valores de y_pred son enteros de 0 a 4.
[ ] predictions.csv tiene una fila por cada imagen de test.
[ ] El team_id es el mismo que en entregas anteriores.
[ ] El enlace al CSV es accesible para el instructor/flujo del leaderboard.
```
