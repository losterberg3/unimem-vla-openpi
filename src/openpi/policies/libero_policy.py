import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 4:
        # Video: (T, C, H, W) → (T, H, W, C)
        if image.shape[1] == 3:
            image = np.transpose(image, (0, 2, 3, 1))
    elif image.shape[0] == 3:
        # Single frame: (C, H, W) → (H, W, C)
        # Use numpy only: einops may probe a broken TensorFlow install (no tf.Tensor) on some arrays.
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        def _mask(val: bool, ref_image: np.ndarray) -> np.ndarray:
            # For video inputs (T, H, W, C), return (T,) so that after batching
            # the mask shape (B, T) matches the image batch prefix (B, T) for jaxtyping.
            if ref_image.ndim == 4:
                return np.full(ref_image.shape[0], val)
            return np.bool_(val)

        # Create inputs dict. Do not change the keys in the dict below.
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
        }
        image_masks = {
            "base_0_rgb": _mask(True, base_image),
            "left_wrist_0_rgb": _mask(True, wrist_image),
        }
        # PI0_FAST requires a third (zero-padded) camera slot; PI0 does not use it.
        if self.model_type == _model.ModelType.PI0_FAST:
            images["right_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["right_wrist_0_rgb"] = _mask(True, np.zeros_like(base_image))
        inputs = {
            "state": data["observation/state"],
            "image": images,
            "image_mask": image_masks,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "labels" in data:
            # One scalar target per sample; shape (1,) so stacking yields [B, 1] for jaxtyping ``Labels = *b v``.
            lab = np.asarray(data["labels"], dtype=np.int32).ravel()
            if lab.size != 1:
                raise ValueError(f"Expected one label per sample, got shape {lab.shape}")
            inputs["labels"] = lab.reshape(1)

        # Key stays "phase_history": it must match the LeRobot dataset column and the
        # RepackTransform config that every event-tracking checkpoint was trained with.
        if "phase_history" in data:
            inputs["phase_history"] = data["phase_history"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        outputs = {
            "actions": np.asarray(data["actions"][:, :, :7]),
        }
        if "event_id" in data:
            outputs["event_id"] = np.asarray(data["event_id"])

        return outputs
