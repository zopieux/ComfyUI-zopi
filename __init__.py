import difflib
from pathlib import Path

import nodes as comfyui_nodes
import folder_paths


class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class EvalPython:
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("OUTPUT",)

    CATEGORY = "utils"
    FUNCTION = "eval_python"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {
                    "multiline": True,
                    "default": "output = input",
                    "tooltip": "Python script; you have access to 'input' and must set the value of 'output'.",
                }),
            },
            "optional": {
                "input": (any, {
                    "forceInput": False,
                    "tooltip": "Any input; may be None",
                }),
            }
        }

    def eval_python(self, expression, input=None):
        globals = {"input": input, "output": None}
        exec(expression, globals)
        output = globals.get("output", None)
        return (output,)


class LoadTensortRTAndCheckpoint:
    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("TensorRT MODEL", "SD MODEL", "CLIP", "VAE", "TensorRT name", "SD name", "Model type")
    OUTPUT_NODE = True

    CATEGORY = "loaders"
    FUNCTION = "load_unet"

    _CKPT_DIR = "checkpoints"
    _TENSORRT_DIR = "tensorrt"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list(cls._TENSORRT_DIR),),}}

    def load_unet(self, unet_name):
        model_type = "sd1.x"
        if "xl" in unet_name.lower():
            model_type = "sdxl_base"

        unet_path = Path(folder_paths.get_full_path(self._TENSORRT_DIR, unet_name))
        if not unet_path.exists():
            raise FileNotFoundError(f"File {unet_path} does not exist")

        name = unet_path.stem.split('_$', 1)[0]
        all_checkpoints = list(folder_paths.get_filename_list(self._CKPT_DIR))
        all_names = [Path(x).stem.lower() for x in all_checkpoints]
        best = find_best_match_with_difflib(name.lower(), all_names)
        idx = all_names.index(best)
        ckpt_name = Path(all_checkpoints[idx]).name

        CheckpointLoaderSimple = hackily_get_node("CheckpointLoaderSimple")
        ckpt, clip, vae = CheckpointLoaderSimple(ckpt_name)
        TensorRTLoader = hackily_get_node("TensorRTLoader")
        unet, = TensorRTLoader(unet_name, model_type)

        return {
            "ui": {
                "texts": [ckpt_name, model_type]
            },
            "result": (unet, ckpt, clip, vae, unet_name, ckpt_name, model_type),
        }


# Cheers to great software design!
def hackily_get_node(name: str):
    kls = comfyui_nodes.NODE_CLASS_MAPPINGS[name]
    inst = kls()
    return getattr(inst, inst.FUNCTION)


def find_best_match_with_difflib(needle, haystack, cutoff=0.0):
    if not haystack:
        return None
    if matches := difflib.get_close_matches(word=needle, possibilities=haystack, n=1, cutoff=cutoff):
        return matches[0]


NODE_CLASS_MAPPINGS = {
    "EvalPython": EvalPython,
    "LoadTensortRTAndCheckpoint": LoadTensortRTAndCheckpoint,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EvalPython": "Eval Python",
    "LoadTensortRTAndCheckpoint": "Load TensortRT + checkpoint + CLIP + VAE",
}
WEB_DIRECTORY = "web"
