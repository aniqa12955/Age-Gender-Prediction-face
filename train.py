import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter

DATASET_PATHS = ["crop_part1", "UTKFace"]
IMG_SIZE = 64

images, ages, genders = [], [], []
skipped = 0

# ===============================
# LOAD DATASET
# ===============================
for DATASET_PATH in DATASET_PATHS:
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ Folder nahi mila: {DATASET_PATH}")
        continue
    print(f"📂 Loading: {DATASET_PATH}")

    for img_name in os.listdir(DATASET_PATH):
        try:
            parts = img_name.split("_")
            if len(parts) < 2:
                skipped += 1
                continue

            age = int(parts[0])
            gender = int(parts[1])

            if not (1 <= age <= 116) or gender not in [0, 1]:
                skipped += 1
                continue

            img_path = os.path.join(DATASET_PATH, img_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            images.append(img)
            ages.append(age)
            genders.append(gender)

        except Exception:
            skipped += 1
            continue

print(f"✅ Loaded: {len(images)} | Skipped: {skipped}")

# ===============================
# AGE BINS
# ===============================
def age_to_class(age):
    if age <= 10:   return 0
    elif age <= 20: return 1
    elif age <= 30: return 2
    elif age <= 40: return 3
    elif age <= 50: return 4
    else:           return 5

age_classes_raw = np.array([age_to_class(a) for a in ages])

print("\n📊 Age class distribution:")
for cls, count in sorted(Counter(age_classes_raw).items()):
    label = ["0-10","11-20","21-30","31-40","41-50","50+"][cls]
    print(f"  Class {cls} ({label}): {count} images")

# ===============================
# ARRAYS
# ===============================
images      = np.array(images, dtype="float32")
genders     = np.array(genders, dtype="float32")
age_classes = to_categorical(age_classes_raw, num_classes=6)

# ===============================
# CLASS WEIGHTS
# ===============================
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(age_classes_raw),
    y=age_classes_raw
)
age_class_weights = dict(enumerate(class_weights_arr))
print("\n⚖️  Age class weights:", age_class_weights)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_age_train, y_age_test, y_gen_train, y_gen_test = train_test_split(
    images, age_classes, genders,
    test_size=0.2, random_state=42, stratify=age_classes_raw
)

print(f"\n🔀 Train: {len(X_train)} | Test: {len(X_test)}")

# ===============================
# CUSTOM DATA GENERATOR
# ===============================
def make_dataset(X, y_age, y_gen, batch_size=32, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, {'age_output': y_age, 'gender_output': y_gen})
    )
    if augment:
        def aug(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, label
        dataset = dataset.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = make_dataset(X_train, y_age_train, y_gen_train, batch_size=32, augment=True)
test_dataset  = make_dataset(X_test,  y_age_test,  y_gen_test,  batch_size=32, augment=False)

# ===============================
# MODEL
# ===============================
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)
x = Dropout(0.25)(x)

x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

age_output    = Dense(6, activation='softmax', name='age_output')(x)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

model = Model(inputs=input_layer, outputs=[age_output, gender_output])
model.summary()

# ===============================
# COMPILE
# ===============================
model.compile(
    optimizer='adam',
    loss={
        'age_output':    'categorical_crossentropy',
        'gender_output': 'binary_crossentropy'
    },
    loss_weights={
        'age_output':    2.0,
        'gender_output': 1.0
    },
    metrics={
        'age_output':    'accuracy',
        'gender_output': 'accuracy'
    }
)

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    ModelCheckpoint(
        "age_gender_best.h5",
        monitor='val_age_output_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_age_output_accuracy',
        patience=8,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_age_output_accuracy',
        factor=0.5,
        patience=4,
        mode='max',
        min_lr=1e-6,
        verbose=1
    )
]

# ===============================
# TRAIN
# ===============================
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=60,
    callbacks=callbacks
)

# ===============================
# SAVE
# ===============================
model.save("age_gender_model.h5")
print("\n✅ Final model saved: age_gender_model.h5")
print("✅ Best model saved:  age_gender_best.h5")