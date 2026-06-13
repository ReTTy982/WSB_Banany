import json
import os
import tempfile
import zipfile

from tensorflow import keras


def remove_legacy_rescaling_arguments(config):
    if isinstance(config, dict):
        if config.get("class_name") == "Rescaling":
            config.get("config", {}).pop("batch_input_shape", None)

        for value in config.values():
            remove_legacy_rescaling_arguments(value)
    elif isinstance(config, list):
        for value in config:
            remove_legacy_rescaling_arguments(value)


def create_compatible_model_copy(model_path):
    temporary_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    temporary_path = temporary_file.name
    temporary_file.close()

    with zipfile.ZipFile(model_path, "r") as source_archive:
        config = json.loads(source_archive.read("config.json"))
        remove_legacy_rescaling_arguments(config)

        with zipfile.ZipFile(temporary_path, "w") as target_archive:
            for archive_entry in source_archive.infolist():
                if archive_entry.filename == "config.json":
                    data = json.dumps(config).encode("utf-8")
                else:
                    data = source_archive.read(archive_entry.filename)

                target_archive.writestr(archive_entry, data)

    return temporary_path


def load_compatible_model(model_path, compile=False):
    try:
        return keras.models.load_model(model_path, compile=compile)
    except ValueError as error:
        if "batch_input_shape" not in str(error):
            raise

    temporary_path = create_compatible_model_copy(model_path)
    try:
        return keras.models.load_model(temporary_path, compile=compile)
    finally:
        os.remove(temporary_path)
