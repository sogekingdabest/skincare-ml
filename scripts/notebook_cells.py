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
    3: r"""
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_labeled_tfrecord(example):
    labeled_tfrecord_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'sex': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'anatom_site_general_challenge': tf.io.FixedLenFeature(
            [],
            tf.int64,
            default_value=-1,
        ),
    }
    example = tf.io.parse_single_example(example, labeled_tfrecord_format)

    image = decode_image(example['image'])
    age = tf.cast(example['age_approx'], tf.float32) / 100.0
    sex = tf.cast(example['sex'], tf.float32)
    site = tf.cast(
        example['anatom_site_general_challenge'],
        tf.float32,
    ) / 7.0

    metadata = tf.stack([age, sex, site])
    target = tf.reshape(
        tf.cast(example['target'], tf.float32),
        [1],
    )
    return (image, metadata), target


def data_augment(inputs, target):
    image, metadata = inputs

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, max_delta=0.01)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Colour transforms can leave the documented input interval.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return (image, metadata), target
""",
    7: r"""
def focal_loss(gamma=2.0, alpha=0.75):
    return tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        alpha=alpha,
        gamma=gamma,
        from_logits=False,
    )
""",
    9: r"""
def build_multimodal_model(img_size=256, num_metadata=3):
    # External image contract: float32 RGB values in [0, 1].
    input_img = layers.Input(
        shape=(img_size, img_size, 3),
        name='input_img',
    )

    # EfficientNetV2 without its built-in preprocessing expects [-1, 1].
    # Keeping this conversion inside the model makes the Keras and TFLite
    # preprocessing contracts identical.
    efficientnet_input = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name='efficientnet_input_rescaling',
    )(input_img)
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        include_preprocessing=False,
        weights='imagenet',
    )
    base_model.trainable = False

    # Keep BatchNormalization statistics in inference mode even after the
    # convolutional layers are partially unfrozen for fine-tuning.
    x_img = base_model(efficientnet_input, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)

    input_meta = layers.Input(
        shape=(num_metadata,),
        name='input_meta',
    )
    x_meta = layers.Dense(16, activation='relu')(input_meta)
    x_meta = layers.BatchNormalization()(x_meta)

    concat = layers.Concatenate()([x_img, x_meta])
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(
        inputs=[input_img, input_meta],
        outputs=output,
    )
    return model, base_model


model, base_model = build_multimodal_model(img_size=256)

preprocessing_probe = model.get_layer('efficientnet_input_rescaling')(
    tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
)
np.testing.assert_allclose(
    preprocessing_probe.numpy(),
    [-1.0, 0.0, 1.0],
    atol=1e-7,
)

print(f"Parámetros totales: {model.count_params():,}")
print(
    "Parámetros entrenables: "
    f"{sum(tf.size(weight).numpy() for weight in model.trainable_weights):,}"
)
""",
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


def count_tfrecord_items(filenames):
    total = 0
    for filename in filenames:
        shard_name = Path(filename).stem
        try:
            total += int(shard_name.rsplit('-', maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f'No se pudo obtener el número de registros de {filename}.'
            ) from error
    return total


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
        dataset = dataset.shuffle(
            2048,
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.repeat()
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
)
test_generator = make_dataset(TEST_FILENAMES, BATCH_SIZE)

NUM_TRAINING_IMAGES = count_tfrecord_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_tfrecord_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_tfrecord_items(TEST_FILENAMES)
STEPS_PER_EPOCH = (
    NUM_TRAINING_IMAGES + BATCH_SIZE - 1
) // BATCH_SIZE

print(f"Imágenes de entrenamiento: {NUM_TRAINING_IMAGES:,}")
print(f"Imágenes de validación: {NUM_VALIDATION_IMAGES:,}")
print(f"Imágenes de test: {NUM_TEST_IMAGES:,}")
print(f"Pasos por época: {STEPS_PER_EPOCH}")
""",
    11: r"""
EPOCHS_FASE_1 = 5

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=STEPS_PER_EPOCH * EPOCHS_FASE_1,
    alpha=0.01,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc', curve='ROC'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)


def validate_batch_contract(dataset, predictive_model, split_name):
    (images, metadata), labels = next(iter(dataset.take(1)))

    tf.debugging.assert_type(images, tf.float32)
    tf.debugging.assert_type(metadata, tf.float32)
    tf.debugging.assert_type(labels, tf.float32)
    tf.debugging.assert_shapes(
        [
            (images, (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            (metadata, (None, 3)),
            (labels, (None, 1)),
        ]
    )
    tf.debugging.assert_all_finite(images, f'{split_name}: imágenes no finitas')
    tf.debugging.assert_all_finite(
        metadata,
        f'{split_name}: metadatos no finitos',
    )
    tf.debugging.assert_all_finite(labels, f'{split_name}: etiquetas no finitas')
    tf.debugging.assert_greater_equal(images, 0.0)
    tf.debugging.assert_less_equal(images, 1.0)
    tf.debugging.assert_equal(
        tf.reduce_all(
            tf.logical_or(tf.equal(labels, 0.0), tf.equal(labels, 1.0))
        ),
        True,
        message=f'{split_name}: se encontraron etiquetas distintas de 0/1',
    )

    predictions = predictive_model([images, metadata], training=False)
    tf.debugging.assert_shapes([(predictions, (None, 1))])
    tf.debugging.assert_all_finite(
        predictions,
        f'{split_name}: predicciones no finitas',
    )
    print(
        f"Preflight {split_name}: OK | "
        f"batch={images.shape[0]} | "
        f"rango=[{tf.reduce_min(images).numpy():.4f}, "
        f"{tf.reduce_max(images).numpy():.4f}]"
    )


validate_batch_contract(train_generator, model, 'training')
validate_batch_contract(val_generator, model, 'validation')

print("Modelo compilado y preflight superado")

os.makedirs('checkpoints', exist_ok=True)

callbacks = [
    ModelCheckpoint(
        'checkpoints/best_model_efficientnet.weights.h5',
        monitor='val_pr_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_pr_auc',
        patience=3,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    ),
]

print("\n" + "=" * 60)
print("FASE 1: ENTRENAMIENTO INICIAL (BACKBONE CONGELADO)")
print("=" * 60)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS_FASE_1,
    callbacks=callbacks,
    verbose=1,
)
""",
    13: r"""
print("\n" + "=" * 60)
print("FASE 2: FINE-TUNING PARCIAL")
print("=" * 60)

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_backbone_layers = [
    layer for layer in base_model.layers if layer.trainable
]
if not trainable_backbone_layers:
    raise RuntimeError('El fine-tuning no dejó capas entrenables en el backbone.')

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5,
        clipnorm=1.0,
    ),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc', curve='ROC'),
        tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)

trainable_parameters = sum(
    int(tf.size(weight).numpy()) for weight in model.trainable_weights
)
print(f"Capas entrenables del backbone: {len(trainable_backbone_layers)}")
print(f"Parámetros entrenables: {trainable_parameters:,}")

EPOCHS_FASE_2 = 7

callbacks_fase2 = [
    ModelCheckpoint(
        'checkpoints/best_model_multimodal.keras',
        monitor='val_pr_auc',
        mode='max',
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor='val_pr_auc',
        patience=4,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    ),
]

history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS_FASE_1 + EPOCHS_FASE_2,
    initial_epoch=EPOCHS_FASE_1,
    callbacks=callbacks_fase2,
    verbose=1,
)
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
if RUN_384_EXPERIMENT:
    best_model_256.save_weights(
        'checkpoints/best_model_multimodal_256.weights.h5'
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
    'preprocessing': {
        'color_order': 'RGB',
        'external_dtype': 'float32',
        'external_range': [0.0, 1.0],
        'backbone_range': [-1.0, 1.0],
        'backbone_rescaling_embedded_in_model': True,
        'efficientnet_include_preprocessing': False,
    },
    'training_256': {
        'batch_size': 64,
        'training_images': NUM_TRAINING_IMAGES,
        'validation_images': NUM_VALIDATION_IMAGES,
        'test_images': NUM_TEST_IMAGES,
        'steps_per_epoch': STEPS_PER_EPOCH,
        'backbone_frozen_phase_1': True,
        'batch_normalization_inference_mode': True,
    },
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
if RUN_384_EXPERIMENT:
    METRICS_384_REPORT = {
        'run': RUN_METADATA,
        'status': 'completed',
        'dataset': {
            'batch_size': 16,
            'training_images': NUM_TRAINING_IMAGES_384,
            'validation_images': NUM_VALIDATION_IMAGES_384,
            'test_images': NUM_TEST_IMAGES_384,
            'steps_per_epoch': STEPS_PER_EPOCH_384,
        },
        'threshold_selection': THRESHOLD_SELECTION_384,
        'keras_test': METRICS_KERAS_384_TEST,
        'artifact': {
            'filename': (
                'checkpoints/best_model_efficientnet_384.weights.h5'
            ),
            'sha256': sha256_file(
                'checkpoints/best_model_efficientnet_384.weights.h5'
            ),
        },
    }
else:
    METRICS_384_REPORT = {
        'run': RUN_METADATA,
        'status': 'skipped',
        'reason': 'RUN_384_EXPERIMENT=False',
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


CANONICAL_CODE_CELLS.update(
    {
        17: r"""
TEST_FILENAMES_256 = list(TEST_FILENAMES)

if RUN_384_EXPERIMENT:
    print("Limpiando memoria para Progressive Resizing...")
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
    TRAINING_FILENAMES_384 = (
        TRAINING_FILENAMES_2020_384 + FILES_2019_384
    )

    assert_disjoint_shards(
        TRAINING_FILENAMES_2020_384,
        VALIDATION_FILENAMES_384,
        TEST_FILENAMES_384,
    )
    if len(FILES_2020_384) != len(FILES_2020):
        raise RuntimeError(
            'Las resoluciones 256 y 384 no contienen el mismo número '
            'de shards 2020.'
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

    NUM_TRAINING_IMAGES_384 = count_tfrecord_items(
        TRAINING_FILENAMES_384
    )
    NUM_VALIDATION_IMAGES_384 = count_tfrecord_items(
        VALIDATION_FILENAMES_384
    )
    NUM_TEST_IMAGES_384 = count_tfrecord_items(TEST_FILENAMES_384)
    STEPS_PER_EPOCH_384 = (
        NUM_TRAINING_IMAGES_384 + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(f"Imágenes 384 de entrenamiento: {NUM_TRAINING_IMAGES_384:,}")
    print(f"Imágenes 384 de validación: {NUM_VALIDATION_IMAGES_384:,}")
    print(f"Imágenes 384 de test: {NUM_TEST_IMAGES_384:,}")
    print(f"Pasos por época (384 px): {STEPS_PER_EPOCH_384}")
else:
    print(
        'Experimento de 384 px omitido. '
        'Configura RUN_384_EXPERIMENT=True para ejecutarlo.'
    )
""",
        19: r"""
if RUN_384_EXPERIMENT:
    print("\n" + "=" * 60)
    print("FASE 3: PROGRESSIVE RESIZING (384x384)")
    print("=" * 60)

    model_384, base_model_384 = build_multimodal_model(img_size=384)
    model_384.load_weights(
        'checkpoints/best_model_multimodal_256.weights.h5'
    )

    base_model_384.trainable = True
    fine_tune_at_384 = len(base_model_384.layers) - 50
    for layer in base_model_384.layers[:fine_tune_at_384]:
        layer.trainable = False
    for layer in base_model_384.layers[fine_tune_at_384:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model_384.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=5e-6,
            clipnorm=1.0,
        ),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )

    validate_batch_contract(
        train_generator_384,
        model_384,
        'training_384',
    )
    validate_batch_contract(
        val_generator_384,
        model_384,
        'validation_384',
    )

    EPOCHS_FASE_3 = 2
    callbacks_phase3 = [
        ModelCheckpoint(
            'checkpoints/best_model_efficientnet_384.weights.h5',
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    ]

    history_phase3 = model_384.fit(
        train_generator_384,
        validation_data=val_generator_384,
        steps_per_epoch=STEPS_PER_EPOCH_384,
        epochs=EPOCHS_FASE_3,
        callbacks=callbacks_phase3,
        verbose=1,
    )
else:
    print('Entrenamiento de 384 px omitido.')
""",
        21: r"""
if RUN_384_EXPERIMENT:
    model_384.load_weights(
        'checkpoints/best_model_efficientnet_384.weights.h5'
    )

    print('Seleccionando el umbral sobre validación (384 px)...')
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
else:
    THRESHOLD_SELECTION_384 = None
    METRICS_KERAS_384_TEST = None
    print('Evaluación de 384 px omitida.')
""",
    }
)
