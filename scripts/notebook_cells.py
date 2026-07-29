"""Canonical evaluation and export cells for the Kaggle notebook."""

CANONICAL_MARKDOWN_SECTIONS = {
    4: """## 3. Datasets y partición a 256 × 256

Los shards de SIIM-ISIC 2020 se barajan con una semilla fija y se dividen en
entrenamiento (70 %), validación para seleccionar el umbral (15 %) y test
independiente (15 %). Los shards de ISIC 2019 se usan solo para entrenamiento.

La separación es por shards. El notebook comprueba los identificadores de
paciente cuando están disponibles y se detiene si encuentra solapamiento. Si
los TFRecords no incluyen ese campo, la limitación queda registrada en las
métricas y el test no debe presentarse como validación clínica externa.
""",
    14: """## 8. Evaluación del modelo de 256 × 256

El umbral se selecciona exclusivamente sobre validación, buscando una
sensibilidad mínima del 85 %. Después se congela y se calculan ROC AUC, average
precision, sensibilidad, especificidad, precisión, NPV, F1, MCC y balanced
accuracy sobre el conjunto de test independiente.

El checkpoint evaluado es exactamente el que posteriormente se exporta a
TensorFlow Lite.
""",
    20: """## 11. Evaluación experimental a 384 × 384

El modelo de 384 píxeles utiliza los mismos índices de shards. Su umbral se
selecciona en validación y sus métricas finales se calculan en test. Los
resultados y el checkpoint permanecen separados de los del modelo móvil de
256 píxeles.
""",
    22: """## 12. Exportación y verificación del modelo de 256 × 256

La exportación carga el mejor checkpoint completo de la fase 2 y genera
`skincare_multimodal_256.tflite` con optimización dinámica de pesos. El TFLite
se ejecuta sobre el mismo test independiente y sus probabilidades se comparan
con las del checkpoint Keras.

La ejecución genera `metrics_256.json`, `metrics_384.json` y `metrics.json` con
splits, umbrales, matrices de confusión, paridad, versiones y hashes SHA-256.
""",
}


CANONICAL_CODE_CELLS = {
    5: r"""
IMAGE_SIZE = [256, 256]
BATCH_SIZE = 64

GCS_PATH_2020 = '/kaggle/input/datasets/cdeotte/melanoma-256x256'
GCS_PATH_2019 = '/kaggle/input/datasets/cdeotte/isic2019-256x256'

FILES_2020 = sorted(tf.io.gfile.glob(GCS_PATH_2020 + '/train*.tfrec'))
FILES_2019 = sorted(tf.io.gfile.glob(GCS_PATH_2019 + '/train*.tfrec'))

if not FILES_2020 or not FILES_2019:
    raise FileNotFoundError(
        'No se encontraron los TFRecords de 256 px. '
        'Adjunta los datasets indicados en docs/DATASETS.md.'
    )


def split_tfrecord_shards(
    filenames,
    validation_fraction=0.15,
    test_fraction=0.15,
    seed=SEED,
):
    # Create deterministic, mutually exclusive shard splits.
    ordered = sorted(filenames)
    if len(ordered) < 3:
        raise ValueError('Se necesitan al menos tres shards para los tres splits.')
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError('Las fracciones de validación y test deben ser positivas.')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('La suma de validación y test debe ser menor que uno.')

    rng = np.random.default_rng(seed)
    shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    training_count = len(shuffled) - validation_count - test_count
    if training_count < 1:
        raise ValueError('La partición no deja shards para entrenamiento.')

    training = sorted(shuffled[:training_count])
    validation = sorted(
        shuffled[training_count:training_count + validation_count]
    )
    test = sorted(shuffled[training_count + validation_count:])
    return training, validation, test


(
    TRAINING_FILENAMES_2020,
    VALIDATION_FILENAMES,
    TEST_FILENAMES,
) = split_tfrecord_shards(FILES_2020)
TRAINING_FILENAMES = TRAINING_FILENAMES_2020 + FILES_2019


def assert_disjoint_shards(training, validation, test):
    split_sets = {
        'training': set(training),
        'validation': set(validation),
        'test': set(test),
    }
    split_names = list(split_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1:]:
            if split_sets[left_name].intersection(split_sets[right_name]):
                raise RuntimeError(
                    f'Solapamiento de shards entre {left_name} y {right_name}.'
                )


assert_disjoint_shards(
    TRAINING_FILENAMES_2020,
    VALIDATION_FILENAMES,
    TEST_FILENAMES,
)


def collect_patient_ids(filenames):
    patient_spec = {
        'patient_id': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }

    def parse_patient_id(example):
        parsed = tf.io.parse_single_example(example, patient_spec)
        return parsed['patient_id']

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(parse_patient_id, num_parallel_calls=AUTO).batch(1024)
    patient_ids = set()
    for batch in dataset:
        patient_ids.update(
            int(raw)
            for raw in batch.numpy()
            if raw >= 0
        )
    return patient_ids


def audit_patient_splits(training, validation, test):
    patient_sets = {
        'training': collect_patient_ids(training),
        'validation': collect_patient_ids(validation),
        'test': collect_patient_ids(test),
    }
    counts = {name: len(values) for name, values in patient_sets.items()}
    if not all(counts.values()):
        return {
            'status': 'patient_id_unavailable',
            'patient_counts': counts,
            'overlap_counts': None,
        }

    overlap_counts = {
        'training_validation': len(
            patient_sets['training'] & patient_sets['validation']
        ),
        'training_test': len(patient_sets['training'] & patient_sets['test']),
        'validation_test': len(
            patient_sets['validation'] & patient_sets['test']
        ),
    }
    if any(overlap_counts.values()):
        raise RuntimeError(
            f'Se detectó solapamiento de pacientes: {overlap_counts}'
        )
    return {
        'status': 'verified_no_overlap',
        'patient_counts': counts,
        'overlap_counts': overlap_counts,
    }


PATIENT_AUDIT_256 = audit_patient_splits(
    TRAINING_FILENAMES,
    VALIDATION_FILENAMES,
    TEST_FILENAMES,
)

print(f"Shards de entrenamiento: {len(TRAINING_FILENAMES)}")
print(f"Shards de validación de umbral: {len(VALIDATION_FILENAMES)}")
print(f"Shards de test independiente: {len(TEST_FILENAMES)}")
print(f"Auditoría de pacientes: {PATIENT_AUDIT_256}")


def make_dataset(filenames, batch_size, training=False, cache=False):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    if training:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048, seed=SEED)
    dataset = dataset.batch(batch_size)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(AUTO)


train_generator = make_dataset(
    TRAINING_FILENAMES,
    BATCH_SIZE,
    training=True,
)
val_generator = make_dataset(
    VALIDATION_FILENAMES,
    BATCH_SIZE,
    cache=True,
)
test_generator = make_dataset(TEST_FILENAMES, BATCH_SIZE)

NUM_TRAINING_IMAGES = len(TRAINING_FILENAMES) * 2071
NUM_VALIDATION_IMAGES = len(VALIDATION_FILENAMES) * 2071
NUM_TEST_IMAGES = len(TEST_FILENAMES) * 2071
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(f"Imágenes de entrenamiento aproximadas: {NUM_TRAINING_IMAGES:,}")
print(f"Imágenes de validación aproximadas: {NUM_VALIDATION_IMAGES:,}")
print(f"Imágenes de test aproximadas: {NUM_TEST_IMAGES:,}")
print(f"Pasos por época: {STEPS_PER_EPOCH}")
""",
    15: r"""
def collect_predictions(predictive_model, dataset):
    labels = []
    probabilities = []
    for (batch_images, batch_meta), batch_labels in dataset:
        predictions = predictive_model.predict(
            [batch_images, batch_meta],
            verbose=0,
        )
        probabilities.extend(predictions.reshape(-1))
        labels.extend(batch_labels.numpy().reshape(-1))
    return np.asarray(labels), np.asarray(probabilities)


def find_threshold_for_minimum_sensitivity(
    y_true,
    y_pred,
    minimum_sensitivity=0.85,
):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    valid_indices = np.where(tpr >= minimum_sensitivity)[0]
    if len(valid_indices):
        selected_index = valid_indices[np.argmin(fpr[valid_indices])]
    else:
        selected_index = int(np.argmax(tpr - fpr))
    return {
        'threshold': float(thresholds[selected_index]),
        'validation_sensitivity': float(tpr[selected_index]),
        'validation_specificity': float(1 - fpr[selected_index]),
        'minimum_sensitivity': float(minimum_sensitivity),
    }


def evaluate_model_comprehensive(
    y_true,
    y_pred,
    threshold,
    model_name,
    split_name,
):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_binary = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    youden_index = sensitivity + specificity - 1
    number_needed_to_diagnose = (
        1 / youden_index if youden_index > 0 else None
    )

    metrics = {
        'model': model_name,
        'split': split_name,
        'sample_count': int(len(y_true)),
        'positive_count': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
        'threshold': float(threshold),
        'auc_roc': float(roc_auc_score(y_true, y_pred)),
        'average_precision': float(average_precision_score(y_true, y_pred)),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(tp / (tp + fp) if (tp + fp) else 0.0),
        'npv': float(tn / (tn + fn) if (tn + fn) else 0.0),
        'f1_score': float(f1_score(y_true, y_pred_binary)),
        'mcc': float(matthews_corrcoef(y_true, y_pred_binary)),
        'balanced_accuracy': float(
            balanced_accuracy_score(y_true, y_pred_binary)
        ),
        'youden_index': float(youden_index),
        'number_needed_to_diagnose': (
            float(number_needed_to_diagnose)
            if number_needed_to_diagnose is not None
            else None
        ),
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        },
    }

    print("\n" + "=" * 60)
    print(f"EVALUACIÓN: {model_name} | split={split_name}")
    print("=" * 60)
    print(f"Muestras: {metrics['sample_count']:,}")
    print(f"Prevalencia: {metrics['prevalence']:.2%}")
    print(f"Umbral congelado: {threshold:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Sensibilidad: {metrics['sensitivity']:.2%}")
    print(f"Especificidad: {metrics['specificity']:.2%}")
    print(f"Precisión: {metrics['precision']:.2%}")
    print(f"NPV: {metrics['npv']:.2%}")
    print(f"F1: {metrics['f1_score']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Matriz de confusión: {metrics['confusion_matrix']}")
    return metrics


best_model_256 = tf.keras.models.load_model(
    'checkpoints/best_model_multimodal.keras',
    compile=False,
)

print('Seleccionando el umbral exclusivamente sobre validación (256 px)...')
y_validation_true_256, y_validation_pred_256 = collect_predictions(
    best_model_256,
    val_generator,
)
THRESHOLD_SELECTION_256 = find_threshold_for_minimum_sensitivity(
    y_validation_true_256,
    y_validation_pred_256,
)
THRESHOLD_256 = THRESHOLD_SELECTION_256['threshold']
print(f"Selección de umbral: {THRESHOLD_SELECTION_256}")

print('Evaluando el checkpoint de 256 px sobre test independiente...')
y_test_true_256, y_test_pred_keras_256 = collect_predictions(
    best_model_256,
    test_generator,
)
METRICS_KERAS_256_TEST = evaluate_model_comprehensive(
    y_test_true_256,
    y_test_pred_keras_256,
    THRESHOLD_256,
    model_name='keras_multimodal_256',
    split_name='test',
)
""",
    17: r"""
print("Limpiando memoria para Progressive Resizing...")
TEST_FILENAMES_256 = list(TEST_FILENAMES)
del train_generator
del val_generator
del test_generator
gc.collect()
K.clear_session()

IMAGE_SIZE = [384, 384]
BATCH_SIZE = 16

GCS_PATH_2020_384 = '/kaggle/input/datasets/cdeotte/melanoma-384x384'
GCS_PATH_2019_384 = '/kaggle/input/datasets/cdeotte/isic2019-384x384'

FILES_2020_384 = sorted(
    tf.io.gfile.glob(GCS_PATH_2020_384 + '/train*.tfrec')
)
FILES_2019_384 = sorted(
    tf.io.gfile.glob(GCS_PATH_2019_384 + '/train*.tfrec')
)

if not FILES_2020_384 or not FILES_2019_384:
    raise FileNotFoundError(
        'No se encontraron los TFRecords de 384 px. '
        'Adjunta los datasets indicados en docs/DATASETS.md.'
    )

(
    TRAINING_FILENAMES_2020_384,
    VALIDATION_FILENAMES_384,
    TEST_FILENAMES_384,
) = split_tfrecord_shards(FILES_2020_384)
TRAINING_FILENAMES_384 = TRAINING_FILENAMES_2020_384 + FILES_2019_384

assert_disjoint_shards(
    TRAINING_FILENAMES_2020_384,
    VALIDATION_FILENAMES_384,
    TEST_FILENAMES_384,
)
if len(FILES_2020_384) != len(FILES_2020):
    raise RuntimeError(
        'Las resoluciones 256 y 384 no contienen el mismo número de shards 2020.'
    )

train_generator_384 = make_dataset(
    TRAINING_FILENAMES_384,
    BATCH_SIZE,
    training=True,
)
val_generator_384 = make_dataset(
    VALIDATION_FILENAMES_384,
    BATCH_SIZE,
)
test_generator_384 = make_dataset(TEST_FILENAMES_384, BATCH_SIZE)

NUM_TRAINING_IMAGES_384 = len(TRAINING_FILENAMES_384) * 2071
NUM_VALIDATION_IMAGES_384 = len(VALIDATION_FILENAMES_384) * 2071
NUM_TEST_IMAGES_384 = len(TEST_FILENAMES_384) * 2071
STEPS_PER_EPOCH_384 = NUM_TRAINING_IMAGES_384 // BATCH_SIZE

print(f"Shards 384 de entrenamiento: {len(TRAINING_FILENAMES_384)}")
print(f"Shards 384 de validación: {len(VALIDATION_FILENAMES_384)}")
print(f"Shards 384 de test: {len(TEST_FILENAMES_384)}")
print(f"Pasos por época (384 px): {STEPS_PER_EPOCH_384}")
""",
    21: r"""
model_384.load_weights(
    'checkpoints/best_model_efficientnet_384.weights.h5'
)

print('Seleccionando el umbral exclusivamente sobre validación (384 px)...')
y_validation_true_384, y_validation_pred_384 = collect_predictions(
    model_384,
    val_generator_384,
)
THRESHOLD_SELECTION_384 = find_threshold_for_minimum_sensitivity(
    y_validation_true_384,
    y_validation_pred_384,
)
THRESHOLD_384 = THRESHOLD_SELECTION_384['threshold']
print(f"Selección de umbral: {THRESHOLD_SELECTION_384}")

print('Evaluando el checkpoint de 384 px sobre test independiente...')
y_test_true_384, y_test_pred_keras_384 = collect_predictions(
    model_384,
    test_generator_384,
)
METRICS_KERAS_384_TEST = evaluate_model_comprehensive(
    y_test_true_384,
    y_test_pred_keras_384,
    THRESHOLD_384,
    model_name='keras_multimodal_384',
    split_name='test',
)
""",
    23: r"""
import hashlib
from datetime import datetime, timezone

print("\n" + "=" * 60)
print("SECCIÓN 12: EXPORTACIÓN Y PARIDAD TFLITE (256 px)")
print("=" * 60)

model_name_keras = 'skincare_model_multimodal.keras'
tflite_filename = 'skincare_multimodal_256.tflite'

final_model = tf.keras.models.load_model(
    'checkpoints/best_model_multimodal.keras',
    compile=False,
)
final_model.save(model_name_keras)
print(f"Modelo Keras guardado: {model_name_keras}")

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(tflite_filename, 'wb') as model_file:
    model_file.write(tflite_model)


def collect_tflite_predictions(model_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_detail = interpreter.get_output_details()[0]
    image_detail = next(
        detail for detail in input_details
        if len(detail.get('shape_signature', detail['shape'])) == 4
    )
    metadata_detail = next(
        detail for detail in input_details
        if len(detail.get('shape_signature', detail['shape'])) == 2
    )

    labels = []
    probabilities = []
    allocated_batch_size = None
    for (batch_images, batch_meta), batch_labels in dataset:
        image_values = batch_images.numpy().astype(image_detail['dtype'])
        metadata_values = batch_meta.numpy().astype(metadata_detail['dtype'])
        batch_size = image_values.shape[0]
        if batch_size != allocated_batch_size:
            interpreter.resize_tensor_input(
                image_detail['index'],
                image_values.shape,
                strict=False,
            )
            interpreter.resize_tensor_input(
                metadata_detail['index'],
                metadata_values.shape,
                strict=False,
            )
            interpreter.allocate_tensors()
            allocated_batch_size = batch_size

        interpreter.set_tensor(image_detail['index'], image_values)
        interpreter.set_tensor(metadata_detail['index'], metadata_values)
        interpreter.invoke()
        probabilities.extend(
            interpreter.get_tensor(output_detail['index']).reshape(-1)
        )
        labels.extend(batch_labels.numpy().reshape(-1))

    return np.asarray(labels), np.asarray(probabilities)


IMAGE_SIZE = [256, 256]
BATCH_SIZE = 64
tflite_test_generator = make_dataset(TEST_FILENAMES_256, BATCH_SIZE)
y_test_true_tflite_256, y_test_pred_tflite_256 = (
    collect_tflite_predictions(tflite_filename, tflite_test_generator)
)
if not np.array_equal(y_test_true_256, y_test_true_tflite_256):
    raise RuntimeError('Keras y TFLite no evaluaron las mismas muestras.')

METRICS_TFLITE_256_TEST = evaluate_model_comprehensive(
    y_test_true_tflite_256,
    y_test_pred_tflite_256,
    THRESHOLD_256,
    model_name='tflite_multimodal_256',
    split_name='test',
)

absolute_errors = np.abs(y_test_pred_keras_256 - y_test_pred_tflite_256)
PARITY_256 = {
    'sample_count': int(len(absolute_errors)),
    'mean_absolute_error': float(absolute_errors.mean()),
    'max_absolute_error': float(absolute_errors.max()),
    'p99_absolute_error': float(np.quantile(absolute_errors, 0.99)),
    'decision_agreement': float(np.mean(
        (y_test_pred_keras_256 >= THRESHOLD_256)
        == (y_test_pred_tflite_256 >= THRESHOLD_256)
    )),
}
PARITY_256['passed'] = bool(
    PARITY_256['mean_absolute_error'] <= 0.005
    and PARITY_256['p99_absolute_error'] <= 0.02
    and PARITY_256['decision_agreement'] >= 0.99
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


RUN_METADATA = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'seed': SEED,
    'python': platform.python_version(),
    'tensorflow': tf.__version__,
    'numpy': np.__version__,
    'split_strategy': {
        'unit': 'tfrecord_shard',
        'training_fraction_2020': 0.70,
        'validation_fraction_2020': 0.15,
        'test_fraction_2020': 0.15,
        'isic_2019_usage': 'training_only',
        'patient_audit_256': PATIENT_AUDIT_256,
        'training_shards_2020': [
            Path(path).name for path in TRAINING_FILENAMES_2020
        ],
        'validation_shards_2020': [
            Path(path).name for path in VALIDATION_FILENAMES
        ],
        'test_shards_2020': [
            Path(path).name for path in TEST_FILENAMES_256
        ],
    },
}

METRICS_256_REPORT = {
    'run': RUN_METADATA,
    'threshold_selection': THRESHOLD_SELECTION_256,
    'keras_test': METRICS_KERAS_256_TEST,
    'tflite_test': METRICS_TFLITE_256_TEST,
    'keras_tflite_parity': PARITY_256,
    'artifact': {
        'filename': tflite_filename,
        'size_bytes': int(os.path.getsize(tflite_filename)),
        'sha256': sha256_file(tflite_filename),
    },
}
METRICS_384_REPORT = {
    'run': RUN_METADATA,
    'threshold_selection': THRESHOLD_SELECTION_384,
    'keras_test': METRICS_KERAS_384_TEST,
    'artifact': {
        'filename': 'checkpoints/best_model_efficientnet_384.weights.h5',
        'sha256': sha256_file(
            'checkpoints/best_model_efficientnet_384.weights.h5'
        ),
    },
}
METRICS_REPORT = {
    'schema_version': 1,
    'model_256': METRICS_256_REPORT,
    'model_384': METRICS_384_REPORT,
}

for output_path, payload in (
    ('metrics_256.json', METRICS_256_REPORT),
    ('metrics_384.json', METRICS_384_REPORT),
    ('metrics.json', METRICS_REPORT),
):
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write('\n')
    print(f"Métricas guardadas: {output_path}")

print(f"TFLite: {tflite_filename}")
print(f"Tamaño: {os.path.getsize(tflite_filename) / (1024 ** 2):.2f} MiB")
print(f"SHA-256: {METRICS_256_REPORT['artifact']['sha256']}")
print(f"Paridad Keras/TFLite: {PARITY_256}")
""",
}
