import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xarm_example() -> dict:
    """Creates a random input example for the Xarm policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
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
        image = np.transpose(image, (1, 2, 0))
    return image


def _mask(val: bool, ref_image: np.ndarray) -> np.ndarray:
    # For video inputs (T, H, W, C), broadcast scalar mask across T frames so
    # that after batching the mask shape [B, T] matches image shape [B, T, H, W, C].
    if ref_image.ndim == 4:
        return np.full(ref_image.shape[0], val)
    return np.bool_(val)


@dataclasses.dataclass(frozen=True)
class XarmInputs(transforms.DataTransformFn):
    """Maps the physical xArm's observation dict to the model's expected format.
    Used for both training and inference — see LiberoInputs in libero_policy.py for
    the sibling class if you're adapting this to a different (non-xArm) robot.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/state"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # PI0/PI05 only use exterior + wrist; no third camera slot needed.
                image_dict = {"base_0_rgb": base_image, "left_wrist_0_rgb": wrist_image}
                image_mask_dict = {"base_0_rgb": _mask(True, base_image), "left_wrist_0_rgb": _mask(True, wrist_image)}
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (_mask(True, base_image), _mask(True, base_image), _mask(True, wrist_image))
                image_dict = dict(zip(names, images, strict=True))
                image_mask_dict = dict(zip(names, image_masks, strict=True))
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
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
class XarmOutputs(transforms.DataTransformFn):
    """Maps the model's outputs back to the xArm-specific format. Inference only."""

    def __call__(self, data: dict) -> dict:
        outputs = {
            "actions": np.asarray(data["actions"][:, :, :7]),
        }
        if "event_id" in data:
            outputs["event_id"] = np.asarray(data["event_id"])

        return outputs
