import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
import platform
import numpy as np
import psutil
import io
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from compress import print_sparsity
from utils import astronet_logger
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from datasets import (
    lazy_load_plasticc_noZ,
    lazy_load_plasticc_wZ,
)
from fetch_models import fetch_model
from metrics import (
    DistributedWeightedLogLoss,
    WeightedLogLoss,
)
from utils import astronet_logger, find_optimal_batch_size


log = astronet_logger(__file__)

# Visualization utilities
plt.rc("font", size=20)
plt.rc("figure", figsize=(15, 3))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class SGEBreakoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=24):
        super(SGEBreakoutCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        hrs = subprocess.run(
            f"qstat -j {os.environ.get('JOB_ID')} | grep 'cpu' | awk '{{print $3}}' | awk -F ':' '{{print $1}}' | awk -F  '=' '{{print $2}}'",
            check=True,
            capture_output=True,
            shell=True,
            text=True,
        ).stdout.strip()

        if int(hrs) > self.threshold:
            log.info("Stopping training...")
            self.model.stop_training = True


class PrintModelSparsity(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        sparsity = print_sparsity(self.model)
        log.info(f"Epoch Start -- Current level of sparsity: {sparsity}")

    def on_epoch_end(self, epoch, logs={}):
        sparsity = print_sparsity(self.model)
        log.info(f"Epoch End -- Current level of sparsity: {sparsity}")


class TimeHistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print(
            f"Epoch: {epoch}, Val/Train loss ratio: {ratio:.2f} -- \n"
            f"val_loss: {logs['val_loss']}, loss: {logs['loss']}"
        )

        if ratio > self.threshold:
            print("Stopping training...")
            self.model.stop_training = True


class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples

    def __display_digits(self, inputs, outputs, ground_truth, epoch, n=10):
        plt.clf()

        plt.yticks([])
        plt.grid(None)
        inputs = np.reshape(inputs, [n, 28, 28])
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.reshape(inputs, [28, 28 * n])
        plt.imshow(inputs)
        plt.xticks([28 * x + 14 for x in range(n)], outputs)
        for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if outputs[i] == ground_truth[i]:
                t.set_color("green")
            else:
                t.set_color("red")
        plt.grid(None)

    def on_epoch_end(self, epoch, logs=None):
        np.random.seed(RANDOM_SEED)
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        X_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(X_test), axis=1)

        self.__display_digits(X_test, predictions, y_test, epoch, n=self.display_freq)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        GIF_PATH = "./animation.gif"
        imageio.mimsave(GIF_PATH, self.images, fps=1)


# ===================== BEGIN INLINED CONSTANTS =====================
# Get the project root directory
PROJECT_ROOT = Path(__file__).absolute().parent

# Set up the working directory
try:
    PROJECT_WORKING_DIRECTORY = Path(os.environ['ASNWD'])
except Exception as e:
    print(f"Environment variable ASNWD not set: {e}.\nUsing project root directory")
    PROJECT_WORKING_DIRECTORY = PROJECT_ROOT

asnwd = PROJECT_WORKING_DIRECTORY  # keep existing alias
DATA_DIR = PROJECT_ROOT / "Data"
PREPROCESS_DIR = PROJECT_ROOT / "Preprocess"
TRAINING_DIR = PROJECT_ROOT / "Training"
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"

# Create directories if they don't exist
for directory in [DATA_DIR, PREPROCESS_DIR, TRAINING_DIR, EVALUATION_DIR]:
    directory.mkdir(exist_ok=True)

SYSTEM = platform.system()
LOCAL_DEBUG = os.environ.get("LOCAL_DEBUG")

# PLASTICC class mappings and weights
PLASTICC_CLASS_MAPPING = {
    90: "SNIa", 67: "SNIa-91bg", 52: "SNIax", 42: "SNII", 62: "SNIbc",
    95: "SLSN-I", 15: "TDE", 64: "KN", 88: "AGN", 92: "RRL", 65: "M-dwarf",
    16: "EB", 53: "Mira", 6: "$\\mu$-Lens-Single"
}

PLASTICC_WEIGHTS_DICT = {
    6: 1 / 18, 15: 1 / 9, 16: 1 / 18, 42: 1 / 18, 52: 1 / 18, 53: 1 / 18,
    62: 1 / 18, 64: 1 / 9, 65: 1 / 18, 67: 1 / 18, 88: 1 / 18, 90: 1 / 18,
    92: 1 / 18, 95: 1 / 18, 99: 1 / 19, 1: 1 / 18, 2: 1 / 18, 3: 1 / 18,
}

# LSST filter definitions
LSST_FILTER_MAP = {
    0: "lsstu", 1: "lsstg", 2: "lsstr", 3: "lssti", 4: "lsstz", 5: "lssty"
}

LSST_PB_WAVELENGTHS = {
    "lsstu": 3685.0, "lsstg": 4802.0, "lsstr": 6231.0,
    "lssti": 7542.0, "lsstz": 8690.0, "lssty": 9736.0,
}

LSST_PB_COLORS = {
    "lsstu": "#984ea3", "lsstg": "#4daf4a", "lsstr": "#e41a1c",
    "lssti": "#377eb8", "lsstz": "#ff7f00", "lssty": "#e3c530",
}

# ZTF filter definitions
ZTF_FILTER_MAP = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

ZTF_FILTER_MAP_COLORS = {
    1: "#4daf4a", 2: "#e41a1c", 3: "#377eb8"
}

ZTF_PB_WAVELENGTHS = {
    "ztfg": 4804.79, "ztfr": 6436.92, "ztfi": 7968.22
}

ZTF_PB_COLORS = {
    "ztfg": "#4daf4a", "ztfr": "#e41a1c", "ztfi": "#377eb8"
}
# ===================== END INLINED CONSTANTS =====================



# Set up logging
try:
    log = astronet_logger(__file__)
    log.info("Running...\n" + "=" * (shutil.get_terminal_size((80, 20))[0]))
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Working Directory: {asnwd}")
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    log = astronet_logger(str(TRAINING_DIR / "train.py"))

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


warnings.filterwarnings("ignore")


class Training(object):
    def __init__(
        self,
        epochs: int,
        dataset: str,
        model: str,
        redshift: bool,
        architecture: str,
        avocado: bool,
        testset: bool,
        fink: bool,
    ):
        self.architecture = architecture
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.redshift = redshift
        self.avocado = avocado
        self.testset = testset
        self.fink = fink

    def __call__(self):
        """Train a given architecture with, or without redshift, on either UGRIZY or GR passbands

        Parameters
        ----------
        epochs: int
            Number of epochs to run training for. If running locally, this should be < 5
        dataset: str
            Which dataset to train on; current options: {plasticc, wisdm_2010, wisdm_2019}
        model: str
            Model name of the best performing hyperparameters run
        redshift: bool
            Include additional information or redshift and redshift_error
        architecture: str
            Which architecture to train on; current options: {atx, t2, tinho}
        avocado: bool
            Run using augmented data generated from `avocado` pacakge
        testset: bool
            Run using homebrewed dataset constructed from PLAsTiCC 'test set'
        fink: bool
            Reduce number of bands from UGRIZY --> GR for ZTF like run.

        Examples
        --------
        >>> params = {
            "epochs": 2,
            "architecture": architecture,
            "dataset": dataset,
            "model": hyperrun,
            "testset": True,
            "redshift": True,
            "fink": None,
            "avocado": None,
        }
        >>> training = Training(**params)
        >>> loss = training.get_wloss
        """

        def build_label():
            UNIXTIMESTAMP = int(time.time())
            try:
                VERSION = (
                    subprocess.check_output(["git", "describe", "--always"])
                    .strip()
                    .decode()
                )
            except Exception:
                from astronet import __version__ as current_version

                VERSION = current_version
            JOB_ID = os.environ.get("JOB_ID")
            LABEL = f"{UNIXTIMESTAMP}-{JOB_ID}-{VERSION}"

            return LABEL

        LABEL = build_label()
        checkpoint_path = asnwd / self.architecture / "models" / self.dataset / "checkpoints" / f"checkpoint-{LABEL}"
        csv_logger_file = asnwd / "logs" / self.architecture / f"training-{LABEL}.log"

        # Create necessary directories
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger_file.parent.mkdir(parents=True, exist_ok=True)

        # Lazy load data
        data_dir = DATA_DIR / "plasticc" / "processed"
        X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
        Z_train = np.load(data_dir / "Z_train.npy", mmap_mode="r")
        y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")

        X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
        Z_test = np.load(data_dir / "Z_test.npy", mmap_mode="r")
        y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")

        # >>> train_ds.element_spec[1].shape
        # TensorShape([14])
        # num_classes = train_ds.element_spec[1].shape.as_list()[0]
        num_classes = y_train.shape[1]

        if self.fink is not None:
            # Take only G, R bands
            X_train = X_train[:, :, 0:3:2]
            X_test = X_test[:, :, 0:3:2]

        log.info(f"{X_train.shape, y_train.shape}")

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)

        BATCH_SIZE = find_optimal_batch_size(num_samples)
        log.info(f"BATCH_SIZE:{BATCH_SIZE}")

        input_shape = (BATCH_SIZE, timesteps, num_features)
        log.info(f"input_shape:{input_shape}")

        drop_remainder = False

        def get_compiled_model_and_data(loss, drop_remainder):

            if self.redshift is not None:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results_with_z.json"
                input_shapes = [input_shape, (BATCH_SIZE, Z_train.shape[1])]

                train_ds = (
                    lazy_load_plasticc_wZ(X_train, Z_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_wZ(X_test, Z_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )

            else:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results.json"
                input_shapes = input_shape

                train_ds = (
                    lazy_load_plasticc_noZ(X_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_noZ(X_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )

            model, event = fetch_model(
                model=self.model,
                hyper_results_file=hyper_results_file,
                input_shapes=input_shapes,
                architecture=self.architecture,
                num_classes=num_classes,
            )

            # We compile our model with a sampled learning rate and any custom metrics
            learning_rate = event["lr"]
            model.compile(
                loss=loss,
                optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
            )

            return model, train_ds, test_ds, event, hyper_results_file

        VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
        log.info(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

        if len(tf.config.list_physical_devices("GPU")) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
            BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
            VALIDATION_BATCH_SIZE = (
                VALIDATION_BATCH_SIZE * strategy.num_replicas_in_sync
            )
            # Open a strategy scope.
            with strategy.scope():
                # If you are using a `Loss` class instead, set reduction to `NONE` so that
                # we can do the reduction afterwards and divide by global batch size.
                loss = DistributedWeightedLogLoss(
                    reduction=tf.keras.losses.Reduction.AUTO,
                    # global_batch_size=BATCH_SIZE,
                )

                # Compute loss that is scaled by global batch size.
                # loss = tf.reduce_sum(loss_obj()) * (1.0 / BATCH_SIZE)

                # If clustering weights (model compression), build_model. Otherwise, T2Model should produce
                # original model. TODO: Include flag for choosing between the two, following run with FINK
                (
                    model,
                    train_ds,
                    test_ds,
                    event,
                    hyper_results_file,
                ) = get_compiled_model_and_data(loss, drop_remainder)
        else:
            loss = WeightedLogLoss()
            (
                model,
                train_ds,
                test_ds,
                event,
                hyper_results_file,
            ) = get_compiled_model_and_data(loss, drop_remainder)

        if "pytest" in sys.modules or SYSTEM == "Darwin":
            NTAKE = 3

            train_ds = train_ds.take(NTAKE)
            test_ds = test_ds.take(NTAKE)

            ind = np.array([x for x in range(NTAKE * BATCH_SIZE)])
            y_test = np.take(y_test, ind, axis=0)

        time_callback = TimeHistoryCallback()

        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            epochs=self.epochs,
            shuffle=True,
            validation_data=test_ds,
            validation_batch_size=VALIDATION_BATCH_SIZE,
            verbose=False,
            callbacks=[
                time_callback,
                #                DetectOverfittingCallback(
                #                    threshold=2
                #                ),
                CSVLogger(
                    csv_logger_file,
                    separator=",",
                    append=False,
                ),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="val_loss",
                    patience=50,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    mode="min",
                    monitor="val_loss",
                    save_best_only=True,
                ),
                ReduceLROnPlateau(
                    cooldown=5,
                    factor=0.1,
                    mode="min",
                    monitor="loss",
                    patience=5,
                    verbose=1,
                ),
            ],
        )

        model.summary(print_fn=logging.info)

        log.info(f"PER EPOCH TIMING: {time_callback.times}")
        log.info(f"AVERAGE EPOCH TIMING: {np.array(time_callback.times).mean()}")

        log.info(f"PERCENT OF RAM USED: {psutil.virtual_memory().percent}")
        log.info(f"RAM USED: {psutil.virtual_memory().active / (1024*1024*1024)}")

        #        with tf.device("/cpu:0"):
        #            try:
        #                print(f"LL-FULL Model Evaluate: {model.evaluate(test_input, y_test, verbose=0, batch_size=X_test.shape[0])[0]}")
        #            except Exception:
        #                print(f"Preventing possible OOM...")

        log.info(
            f"LL-BATCHED-32 Model Evaluate: {model.evaluate(test_ds, verbose=0)[0]}"
        )
        log.info(
            f"LL-BATCHED-OP Model Evaluate: {model.evaluate(test_ds, verbose=0, batch_size=VALIDATION_BATCH_SIZE)[0]}"
        )

        if drop_remainder:
            ind = np.array(
                [x for x in range((y_test.shape[0] // BATCH_SIZE) * BATCH_SIZE)]
            )
            y_test = np.take(y_test, ind, axis=0)

        y_preds = model.predict(test_ds)

        log.info(f"{y_preds.shape}, {type(y_preds)}")

        WLOSS = loss(y_test, y_preds).numpy()
        log.info(f"LL-Test Model Predictions: {WLOSS:.8f}")
        if "pytest" in sys.modules:
            return WLOSS

        LABEL = (
            "wZ-" + LABEL if self.redshift else "noZ-" + LABEL
        )  # Prepend whether redshift was used or not
        LABEL = (
            "GR-" + LABEL if self.fink else "UGRIZY-" + LABEL
        )  # Prepend which filters have been used in training
        LABEL += f"-LL{WLOSS:.3f}"  # Append loss score

        if SYSTEM != "Darwin":
            model.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}"
            )
            model.save_weights(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/weights/weights-{LABEL}"
            )

        if X_test.shape[0] < 10000:
            batch_size = X_test.shape[0]  # Use all samples in test set to evaluate
        else:
            # Otherwise potential OOM Error may occur loading too many into memory at once
            batch_size = (
                int(VALIDATION_BATCH_SIZE / strategy.num_replicas_in_sync)
                if len(tf.config.list_physical_devices("GPU")) > 1
                else VALIDATION_BATCH_SIZE
            )
            log.info(f"EVALUATE VALIDATION_BATCH_SIZE : {batch_size}")

        event["hypername"] = event["name"]
        event["name"] = f"{LABEL}"

        event["z-redshift"] = self.redshift
        event["avocado"] = self.avocado
        event["testset"] = self.testset
        event["fink"] = self.fink

        event["num_classes"] = num_classes
        event["model_evaluate_on_test_acc"] = model.evaluate(
            test_ds, verbose=0, batch_size=batch_size
        )[1]
        event["model_evaluate_on_test_loss"] = model.evaluate(
            test_ds, verbose=0, batch_size=batch_size
        )[0]
        event["model_prediction_on_test"] = loss(y_test, y_preds).numpy()

        y_test = np.argmax(y_test, axis=1)
        y_preds = np.argmax(y_preds, axis=1)

        event["model_predict_precision_score"] = precision_score(
            y_test, y_preds, average="macro"
        )
        event["model_predict_recall_score"] = recall_score(
            y_test, y_preds, average="macro"
        )

        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            event["{}".format(key)] = value

        learning_rate = event["lr"]
        del event["lr"]

        if self.redshift is not None:
            train_results_file = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results_with_z.json"
        else:
            train_results_file = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results.json"

        with open(train_results_file) as jf:
            data = json.load(jf)
            # print(data)

            previous_results = data["training_result"]
            # appending data to optuna_result
            # print(previous_results)
            previous_results.append(event)
            # print(previous_results)
            # print(data)

        if SYSTEM != "Darwin":
            with open(train_results_file, "w") as rf:
                json.dump(data, rf, sort_keys=True, indent=4)

        if len(tf.config.list_physical_devices("GPU")) < 2 and SYSTEM != "Darwin":
            # PRUNE
            import tensorflow_model_optimization as tfmot

            # Helper function uses `prune_low_magnitude` to make only the
            # Dense layers train with pruning.
            def apply_pruning_to_dense(layer):
                layer_name = layer.__class__.__name__
                # prunable_layers = ["ConvEmbedding", "TransformerBlock", "ClusterWeights"]
                # if layer_name in prunable_layers:
                if isinstance(layer, tfmot.sparsity.keras.PrunableLayer):
                    log.info(f"Pruning {layer_name}")
                    return tfmot.sparsity.keras.prune_low_magnitude(layer)
                return layer

            # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
            # to the layers of the model.
            model_for_pruning = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense,
            )

            model_for_pruning.summary(print_fn=log.info)

            log_dir = f"{asnwd}/logs/{self.architecture}"

            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
                PrintModelSparsity(),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="val_loss",
                    patience=25,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]

            model_for_pruning.compile(
                loss=loss,
                optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
            )

            model_for_pruning.fit(
                train_ds,
                callbacks=callbacks,
                epochs=100,
            )

            model_for_pruning.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}-PRUNED",
                include_optimizer=True,
            )

            model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
            model_for_export.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}-PRUNED-STRIPPED",
                include_optimizer=True,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process named model")

    parser.add_argument(
        "-a", "--architecture", default="tinho", help="Which architecture to train on"
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="wisdm_2010",
        help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'",
    )

    parser.add_argument(
        "-e", "--epochs", default=20, help="How many epochs to run training for"
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>",
    )

    parser.add_argument(
        "-z",
        "--redshift",
        default=None,
        help="Whether to include redshift features or not",
    )

    parser.add_argument(
        "-A",
        "--avocado",
        default=None,
        help="Train using avocado augmented plasticc data",
    )

    parser.add_argument(
        "-t",
        "--testset",
        default=None,
        help="Train using PLAsTiCC test data for representative test",
    )

    parser.add_argument(
        "-f",
        "--fink",
        default=None,
        help="Train using PLAsTiCC but only g and r bands for FINK",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    architecture = args.architecture
    dataset = args.dataset
    EPOCHS = int(args.epochs)
    model = args.model

    avocado = args.avocado
    if avocado is not None:
        avocado = True

    testset = args.testset
    if testset is not None:
        testset = True

    redshift = args.redshift
    if redshift is not None:
        redshift = True

    fink = args.fink
    if fink is not None:
        fink = True

    training = Training(
        epochs=EPOCHS,
        architecture=architecture,
        dataset=dataset,
        model=model,
        redshift=redshift,
        avocado=avocado,
        testset=testset,
        fink=fink,
    )
    if dataset in ["WalkvsRun", "NetFlow"]:
        # WalkvsRun and NetFlow causes OOM errors on GPU, run on CPU instead
        with tf.device("/cpu:0"):
            print(f"{dataset} causes OOM errors on GPU. Running on CPU...")
            training()
    else:
        print("Running on GPU...")
        training()
