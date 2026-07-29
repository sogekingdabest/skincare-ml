"""Build the portfolio-ready Kaggle training notebook.

The script repairs legacy text encoding, removes execution state, refreshes the
documented narrative and installs the canonical data-splitting, evaluation and
export cells. It never executes the notebook.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from notebook_cells import CANONICAL_CODE_CELLS, CANONICAL_MARKDOWN_SECTIONS


MARKDOWN_SECTIONS = {
    0: """# SkinCare AI — Clasificación multimodal de lesiones cutáneas

Este notebook documenta el entrenamiento y la exportación de un clasificador
binario basado en **EfficientNetV2-B0**. El modelo combina una imagen
dermatoscópica con tres variables de contexto: edad aproximada, sexo codificado
y localización anatómica.

> **Aviso de seguridad:** este proyecto es educativo y experimental. No es un
> dispositivo médico, no ofrece un diagnóstico y no sustituye la evaluación de
> un profesional sanitario.

## Objetivo y alcance

El flujo cubre:

1. lectura de TFRecords y preprocesamiento multimodal;
2. entrenamiento inicial a 256 × 256 píxeles;
3. fine-tuning parcial de EfficientNetV2-B0;
4. evaluación con métricas adecuadas para clases desbalanceadas;
5. un experimento independiente de *progressive resizing* a 384 × 384;
6. exportación del mejor checkpoint de 256 × 256 a TensorFlow Lite.

Kaggle es el entorno de referencia. Las versiones exactas de los datasets, sus
licencias y las limitaciones del modelo se documentan en
[`DATASETS.md`](../docs/DATASETS.md) y
[`MODEL_CARD.md`](../docs/MODEL_CARD.md).

## 1. Entorno y reproducibilidad

Se importan las dependencias, se registra el entorno de ejecución y se fijan
semillas para Python, NumPy y TensorFlow. Una misma semilla reduce variaciones,
pero no garantiza resultados idénticos entre versiones de CUDA, cuDNN,
TensorFlow o hardware distinto.
""",
    2: """## 2. Lectura, normalización y aumento de datos

Cada TFRecord contiene la imagen, la etiqueta binaria y tres metadatos. Las
imágenes JPEG se convierten a `float32` y se escalan al intervalo `[0, 1]`.
Los metadatos se transforman en un vector de tres componentes.

El aumento de datos se aplica únicamente a la imagen: reflexiones, pequeños
cambios de tono, saturación, contraste y brillo. El vector de metadatos no se
modifica.
""",
    4: """## 3. Datasets y partición a 256 × 256

El entrenamiento usa TFRecords de SIIM-ISIC 2020 e ISIC 2019 adjuntos al
notebook de Kaggle. Los shards de 2020 se ordenan y se dividen 80/20; todos los
shards de 2019 se añaden al entrenamiento.

Esta es una partición por shards. Antes de interpretar las métricas como una
estimación definitiva de generalización debe verificarse explícitamente que no
exista solapamiento de pacientes entre entrenamiento y validación.
""",
    6: """## 4. Función de pérdida y métricas

El melanoma es la clase minoritaria, por lo que la exactitud global puede ser
engañosa. Se utiliza *focal loss* para aumentar el peso de los ejemplos
difíciles y se registran ROC AUC, PR AUC, precisión y sensibilidad.

La selección del mejor checkpoint se basa en `val_pr_auc`, una métrica más
informativa que `accuracy` cuando las clases están fuertemente desbalanceadas.
""",
    8: """## 5. Arquitectura multimodal

La rama visual utiliza EfficientNetV2-B0 preentrenada con ImageNet. Sus
características se combinan con una pequeña rama densa para los metadatos.
Después de la fusión, una capa sigmoide produce una puntuación binaria.

**Contrato de entrada:**

- imagen RGB `float32`: `[batch, 256, 256, 3]`, escalada a `[0, 1]`;
- metadatos `float32`: `[batch, 3]`;
- salida sigmoide: `[batch, 1]`.
""",
    10: """## 6. Fase 1 — Entrenamiento inicial

La primera fase entrena el modelo con Adam y un *learning rate* con
`CosineDecay`. Se guarda el checkpoint con mejor `val_pr_auc` y se aplica
*early stopping* para evitar continuar cuando la validación deja de mejorar.
""",
    12: """## 7. Fase 2 — Fine-tuning

Se habilita el backbone y se mantienen congeladas sus capas iniciales. Las
últimas 50 capas se ajustan con un *learning rate* menor para adaptar las
representaciones visuales sin destruir de forma brusca los pesos preentrenados.

El mejor modelo completo de esta fase se guarda como
`checkpoints/best_model_multimodal.keras`. Este es el checkpoint utilizado por
la exportación móvil al final del notebook.
""",
    14: """## 8. Evaluación del modelo de 256 × 256

El umbral de decisión se selecciona sobre la curva ROC buscando una
sensibilidad mínima del 85 %. Después se calculan ROC AUC, average precision,
sensibilidad, especificidad, precisión, NPV, F1, MCC y balanced accuracy.

El resultado registrado de referencia está en
[`MODEL_CARD.md`](../docs/MODEL_CARD.md). Es una evaluación interna sobre el
split de validación, no una validación clínica ni externa.
""",
    16: """## 9. Experimento de *progressive resizing* a 384 × 384

Esta sección crea un pipeline independiente con imágenes de mayor resolución y
un batch menor. Se libera memoria antes de construir el nuevo modelo.

La fase de 384 × 384 es experimental: parte del checkpoint de pesos de la fase
inicial y **no** es el modelo exportado a TensorFlow Lite por este notebook.
Mantener esta frontera explícita evita atribuir sus resultados al artefacto
móvil de 256 × 256.
""",
    18: """## 10. Entrenamiento experimental a 384 × 384

Se construye una nueva instancia de la arquitectura, se cargan los pesos
compatibles de la fase inicial y se ajustan las últimas capas con un learning
rate reducido. Su checkpoint se guarda de forma separada.
""",
    20: """## 11. Evaluación experimental a 384 × 384

La evaluación repite el mismo criterio de sensibilidad mínima y el mismo
conjunto de métricas. Estos resultados deben compararse con los de 256 × 256
solo después de verificar que ambos pipelines usan exactamente el mismo split
y contrato de metadatos.
""",
    22: """## 12. Exportación del modelo de 256 × 256

La exportación carga `best_model_multimodal.keras`, el mejor checkpoint completo
de la fase 2, y genera `skincare_multimodal_256.tflite` con optimización
dinámica de pesos.

Este artefacto requiere dos entradas y no es intercambiable con el modelo
single-input de 224 × 224 que actualmente consume la aplicación Android. El
procedimiento de paridad necesario antes de una integración se describe en
[`ANDROID_INTEGRATION.md`](../docs/ANDROID_INTEGRATION.md).
""",
}

MARKDOWN_SECTIONS.update(CANONICAL_MARKDOWN_SECTIONS)


FINAL_MARKDOWN = """## Artefactos y trazabilidad

Una ejecución completa debe conservar fuera de Git:

- el checkpoint `.keras`;
- la exportación `.tflite`;
- el identificador de versión del notebook de Kaggle;
- las versiones exactas del entorno y de los datasets;
- los hashes SHA-256 de los artefactos;
- las métricas y el umbral obtenidos en esa misma ejecución.

Los binarios aprobados se publican como *release assets* y se registran en
[`artifacts/manifest.json`](../artifacts/manifest.json). El repositorio no
incluye imágenes médicas ni credenciales de Kaggle.
"""


def repair_mojibake(text: str) -> str:
    """Repair UTF-8 bytes that were previously decoded as Latin-1."""

    previous = None
    while previous != text:
        previous = text

        def replace_pair(match: re.Match[str]) -> str:
            value = match.group(0)
            for encoding in ("cp1252", "latin-1"):
                try:
                    return value.encode(encoding).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue
            return value

        text = re.sub(r"\u00c3.", replace_pair, text)
    return text


def source_lines(text: str) -> list[str]:
    """Return nbformat-compatible source lines."""

    return text.strip().splitlines(keepends=True)


def replace_once(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise ValueError(f"Expected notebook fragment not found: {old!r}")
    return source.replace(old, new, 1)


def prepare_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    cells = notebook.get("cells", [])
    if len(cells) not in (24, 25):
        raise ValueError(f"Expected 24 or 25 cells, found {len(cells)}")

    for cell in cells:
        source = repair_mojibake("".join(cell.get("source", [])))
        cell["source"] = source_lines(source)
        cell["metadata"] = {
            key: value
            for key, value in cell.get("metadata", {}).items()
            if key == "tags"
        }
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    for index, markdown in MARKDOWN_SECTIONS.items():
        if cells[index].get("cell_type") != "markdown":
            raise ValueError(f"Cell {index + 1} is not Markdown")
        cells[index]["source"] = source_lines(markdown)

    environment = "".join(cells[1]["source"])
    environment = replace_once(
        environment,
        "import warnings\n",
        "import warnings\nimport platform\nimport random\n",
    )
    environment = replace_once(
        environment,
        "warnings.filterwarnings('ignore')\n",
        (
            "warnings.filterwarnings('ignore')\n\n"
            'print(f"Python: {platform.python_version()}")\n'
            'print(f"TensorFlow: {tf.__version__}")\n'
            'print(f"NumPy: {np.__version__}")\n'
        ),
    )
    environment = replace_once(
        environment,
        "SEED = 42\nnp.random.seed(SEED)\n",
        "SEED = 42\nrandom.seed(SEED)\nnp.random.seed(SEED)\n",
    )
    cells[1]["source"] = source_lines(environment)

    data_256 = "".join(cells[5]["source"])
    if "split_tfrecord_shards" not in data_256:
        data_256 = replace_once(
            data_256,
            "FILES_2019 = tf.io.gfile.glob(GCS_PATH_2019 + '/train*.tfrec')\n",
            (
                "FILES_2019 = tf.io.gfile.glob(GCS_PATH_2019 + '/train*.tfrec')\n\n"
                "if not FILES_2020 or not FILES_2019:\n"
                "    raise FileNotFoundError(\n"
                "        'No se encontraron los TFRecords de 256 px. '\n"
                "        'Adjunta los datasets indicados en docs/DATASETS.md.'\n"
                "    )\n"
            ),
        )
    cells[5]["source"] = source_lines(data_256)

    fine_tuning = "".join(cells[13]["source"])
    fine_tuning = fine_tuning.replace(
        "# Recompilamos con un Learning Rate MUCHÍSIMO más bajo (1e-5 en lugar de 1e-4)",
        "# Recompilamos con un learning rate reducido para el fine-tuning",
    )
    fine_tuning = fine_tuning.replace(
        "# Entrenamos otras 5 épocas",
        "# Entrenamos hasta siete épocas adicionales",
    )
    fine_tuning = fine_tuning.replace(
        "# Nuestro paracaídas de seguridad",
        "# Detenemos el entrenamiento cuando deja de mejorar val_pr_auc",
    )
    fine_tuning = fine_tuning.replace(
        "patience=4, # Le damos un margen de 4 épocas de paciencia",
        "patience=4,",
    )
    cells[13]["source"] = source_lines(fine_tuning)

    data_384 = "".join(cells[17]["source"])
    if "split_tfrecord_shards" not in data_384:
        data_384 = replace_once(
            data_384,
            "FILES_2019_384 = tf.io.gfile.glob(GCS_PATH_2019_384 + '/train*.tfrec')\n",
            (
                "FILES_2019_384 = tf.io.gfile.glob(GCS_PATH_2019_384 + '/train*.tfrec')\n\n"
                "if not FILES_2020_384 or not FILES_2019_384:\n"
                "    raise FileNotFoundError(\n"
                "        'No se encontraron los TFRecords de 384 px. '\n"
                "        'Adjunta los datasets indicados en docs/DATASETS.md.'\n"
                "    )\n"
            ),
        )
    cells[17]["source"] = source_lines(data_384)

    evaluation_384 = "".join(cells[21]["source"])
    replacements = {
        "# 1. REINICIAMOS LAS LISTAS (Esto soluciona tu error)": "# Reiniciamos los acumuladores para la evaluación de 384 px",
        "# 2. ITERAMOS SOBRE EL NUEVO GENERADOR DE 384": "# Recorremos el generador multimodal de validación",
        "# Como es multimodal, desempaquetamos (imágenes, metadatos)": "# Desempaquetamos las imágenes y los metadatos",
        "# Pasamos AMBAS entradas al modelo": "# Ejecutamos ambas entradas del modelo",
        "# Ahora sí podemos usar extend porque y_pred e y_true vuelven a ser listas": "# Acumulamos probabilidades y etiquetas",
        "# 3. CONVERTIMOS A NUMPY AL FINAL": "# Convertimos los acumuladores a NumPy",
        "# 4. LLAMADA A TU FUNCIÓN DE EVALUACIÓN COMPREHENSIVA\n# (Asegúrate de pasarle los arrays calculados)": "# Aplicamos la misma evaluación utilizada en la resolución de 256 px",
    }
    for old, new in replacements.items():
        evaluation_384 = evaluation_384.replace(old, new)
    cells[21]["source"] = source_lines(evaluation_384)

    for index, source in CANONICAL_CODE_CELLS.items():
        if cells[index].get("cell_type") != "code":
            raise ValueError(f"Cell {index + 1} is not code")
        cells[index]["source"] = source_lines(source)

    if len(cells) == 24:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {"tags": ["portfolio-summary"]},
                "source": source_lines(FINAL_MARKDOWN),
            }
        )
    else:
        cells[24]["cell_type"] = "markdown"
        cells[24]["metadata"] = {"tags": ["portfolio-summary"]}
        cells[24]["source"] = source_lines(FINAL_MARKDOWN)
        cells[24].pop("outputs", None)
        cells[24].pop("execution_count", None)

    notebook["cells"] = cells
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        type=Path,
        nargs="?",
        default=Path("notebooks/skincare_training.ipynb"),
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    prepared = prepare_notebook(notebook)
    notebook_path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Prepared {notebook_path}")


if __name__ == "__main__":
    main()
