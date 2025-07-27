import ast
import difflib
from typing import Optional, Union, List
from pathlib import Path
import traceback

import nodes as comfyui_nodes
import folder_paths
from aiohttp import web
from server import PromptServer

MAX_EVAL_OUTPUTS = 10

COMFY_TYPES = {
    "str": "STRING",
    "int": "INT",
    "float": "FLOAT",
    "bool": "BOOLEAN",
}


def get_ast_assignments(python_expr: str) -> list[tuple[str, str]]:
    tree = ast.parse(python_expr)
    typed_assignments = []

    def _get_annotation_value(annotation_node):
        if isinstance(annotation_node, ast.Name) and annotation_node.id in COMFY_TYPES:
            return COMFY_TYPES[annotation_node.id]
        elif isinstance(annotation_node, ast.Constant):  # Python 3.8+
            return annotation_node.value
        return None

    def _process_statement_node(node):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                type_hint = _get_annotation_value(node.annotation)
                if type_hint is not None:
                    typed_assignments.append((var_name, type_hint))
        elif isinstance(node, ast.For):
            for sub_node in node.body:
                _process_statement_node(sub_node)
            if node.orelse:
                for sub_node in node.orelse:
                    _process_statement_node(sub_node)
        elif isinstance(node, ast.While):
            for sub_node in node.body:
                _process_statement_node(sub_node)
            if node.orelse:
                for sub_node in node.orelse:
                    _process_statement_node(sub_node)
        elif isinstance(node, ast.If):
            for sub_node in node.body:
                _process_statement_node(sub_node)
            if node.orelse:
                for sub_node in node.orelse:
                    _process_statement_node(sub_node)
        elif isinstance(node, ast.With):
            for sub_node in node.body:
                _process_statement_node(sub_node)
        elif isinstance(node, ast.Try):
            for sub_node in node.body:
                _process_statement_node(sub_node)
            for handler in node.handlers:
                for sub_node in handler.body:
                    _process_statement_node(sub_node)
            if node.orelse:
                for sub_node in node.orelse:
                    _process_statement_node(sub_node)
            if node.finalbody:
                for sub_node in node.finalbody:
                    _process_statement_node(sub_node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pass

    for node in tree.body:
        _process_statement_node(node)

    return typed_assignments


class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class FlexibleOptionalInputType(dict):

    def __init__(self, type, data: Union[dict, None] = None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


def do_the_eval(expression: str, inputs):
    typed_assignments = get_ast_assignments(expression)[:MAX_EVAL_OUTPUTS]
    input_list = [v for k, v in sorted(inputs.items(), key=lambda x: x[0])]
    outputs = {o: None for o, _ in typed_assignments}
    context = {"input": input_list, **inputs, **outputs}
    try:
        exec(expression, context)
    except:
        traceback.print_exc()
        pass
    return [context[k] for k, _ in typed_assignments]


@PromptServer.instance.routes.post("/zopi/eval_python/types")
async def eval_python(request):
    data = await request.json()
    expression = data["expression"]
    try:
        outputs = get_ast_assignments(expression)[:MAX_EVAL_OUTPUTS]
    except:
        return web.HTTPBadRequest()
    return web.json_response({"outputs": outputs})


class EvalPython:
    RETURN_TYPES = tuple(any_type for i in range(MAX_EVAL_OUTPUTS))
    RETURN_NAMES = tuple(f"output_{i:02d}" for i in range(MAX_EVAL_OUTPUTS))

    CATEGORY = "utils"
    FUNCTION = "eval_python"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {
                    "multiline":
                        True,
                    "default":
                        """output: str = str(input[0])""",
                    "tooltip":
                        f"Python script; you have access to 'input[x]' and must assign 1 to {MAX_EVAL_OUTPUTS} type-annotated variables, of any name.",
                }),
            },
            "optional": FlexibleOptionalInputType(any_type),
        }

    def eval_python(self, expression, **kwargs):
        inputs = {k: v for k, v in kwargs.items() if k.startswith("input[")}
        outputs = do_the_eval(expression, inputs)
        return tuple(outputs)


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
