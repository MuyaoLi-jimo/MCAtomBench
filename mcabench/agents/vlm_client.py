from openai import OpenAI
from pathlib import Path
import pathlib
from typing import Union,Literal
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import requests
import io
import math
import torch
from transformers import AutoProcessor,AutoModelForCausalLM,AutoModelForImageTextToText
from transformers import AutoTokenizer

class VlMClient:
    def __init__(self, api_key,base_url,temperature,max_tokens,
                 model_path,tokenizer_path="",
                 **kwargs):
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.use_vllm = True
        
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,  
                trust_remote_code=True,
            )
        
        if base_url:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            models = self.client.models.list()
            self.model_name = models.data[0].id
        else:
            self.use_vllm = False
            model_path = model_path.lower().replace('-','_')
            processor_config = dict(
                do_rescale=False,
                patch_size=14,
                vision_feature_select_strategy="default"
            )
            model_kwargs = dict(
                torch_dtype="auto"
            )
            if 'qwen2_vl' in model_path:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = torch.bfloat16
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, **processor_config)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, trust_remote_code=True,
                                                                     device_map="auto", **model_kwargs) 
            
        self.processor_wrapper = None
        
    def set_processor_wrapper(self,model_name:str=None):
        if not model_name:
            model_name = self.model_name
        self.processor_wrapper = ProcessorWrapper(model_name=model_name, use_vllm=self.use_vllm)
        
    def generate(self,messages:list,verbos:bool=False,if_token_ids=False):
        
        open_logprobs = False
        if verbos:
            open_logprobs = True
        content = ""
        if self.use_vllm:
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                logprobs = open_logprobs,
                extra_body = {"skip_special_tokens":False}
            )
            content = chat_completion.choices[0].message.content
            if if_token_ids:
                outputs = self.tokenizer(content)["input_ids"]
            else:
                outputs = content
        else:
            images = []
            for message in messages:
                for content in message["content"]:
                    if content["type"]=="image":
                        image = encode_image_to_pil(content["image"])
                        images.append(image)
                        
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt], images=images, padding=True, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens-inputs.input_ids.shape[-1],temperature=self.temperature)
            if if_token_ids:
                outputs = [
                    output_ids[len(input_ids) :].tolist()
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ][0]
            else:
                raise AssertionError("DO NOT FINISH")
        return outputs,content


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,max_ratio:int
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(image: Image.Image,  factor: int, min_pixels: int, max_pixels: int,max_ratio:int) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
    image = image.resize((resized_width, resized_height))
    return image

def pil2base64(image):
    """强制中间结果为jpeg""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_image_to_pil(image_input):
    if isinstance(image_input, (str, Path)): 
        try:
            img = Image.open(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except IOError:
            raise ValueError("Could not open the image file. Check the path and file format.")
    elif isinstance(image_input, np.ndarray):
        try:
            img = Image.fromarray(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except TypeError:
            raise ValueError("Numpy array is not in an appropriate format to convert to an image.")
    elif isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        return image_input
    else:
        raise TypeError("Unsupported image input type. Supported types are str, pathlib.Path, numpy.ndarray, and PIL.Image.")
    

def encode_image_to_base64(image:Union[str,pathlib.PosixPath,Image.Image,np.ndarray], format='JPEG') -> str:
    """Encode an image to base64 format, supports URL, numpy array, and PIL.Image."""

    # Case 1: If the input is a URL (str)
    image_encode = None
    if isinstance(image, str) and image[:4]=="http":
        try:
            response = requests.get(image)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve the image from the URL: {e}")
    elif isinstance(image, str) and image[0]=='/':
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    elif isinstance(image,pathlib.PosixPath):
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    # Case 3: If the input is a numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

def get_suffix(image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = 'jpeg'
    elif isinstance(image,str):
        image_suffix = image.split(".")[-1]
    elif isinstance(image,pathlib.PosixPath):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def translate_cv2(image: Union[str, pathlib.PosixPath, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array (PIL is in RGB)
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Check if the NumPy array is in RGB format and has three channels
        if image.shape[2] == 3:  # Only for color images
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image  # No conversion needed for grayscale images
    elif isinstance(image, (str, pathlib.PosixPath)):
        # Read the image using cv2 (assumes BGR format)
        cv2_image = cv2.imread(str(image))  # Convert PosixPath to string if necessary
        if cv2_image is None:
            raise ValueError(f"The image path is incorrect or the file is not accessible: {image}")
    else:
        raise ValueError("Unsupported image format or path type")
    
    return cv2_image

    

class ProcessorWrapper:
    def __init__(self, model_name= "qwen2_vl",use_vllm=True):
        self.use_vllm = use_vllm
        self.model_name = model_name.replace("-","_")
        self.image_factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = 1024 * 28 * 28  #16384 * 28 * 28
        self.max_ratio = 200

    def get_image_message(self,source_data):
        if self.use_vllm:
            image_suffix = get_suffix(source_data)
            image_message = {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"},
                }
        else:
            image_message = {
                "type": "image",
                "image": source_data,
            }
        return image_message

    def create_message_vllm(self,
                            role:Literal["user","assistant"]="user",
                            input_type:Literal["image","text"]="image",
                            image:Union[list,str,pathlib.PosixPath,np.ndarray,Image.Image]=None,
                            prompt:Union[list,str]="",):
        if role not in {"user","assistant"}:
            raise ValueError(f"a invalid role {role}")
        if isinstance(prompt,str):
            prompt = [prompt]
        message = {
            "role": role,
            "content": [],
        }
        if input_type=="image":
            if not isinstance(image,list):
                image = [image]
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}\n"
                })
                if idx < len(image):
                    message["content"].append(self.get_image_message(image[idx]))
            for idx in range(len(prompt), len(image)):
                message["content"].append(self.get_image_message(image[idx])) 
        else:
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}\n"
                })
        return message

    
    def create_text_input(self,conversations:list):
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        return text_prompt
    
    def create_image_input(self,image_pixels=None,image_path:str=""):
        image = image_pixels
        if image_path:
            image = Image.open(image_path)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype('uint8'))
        if "qwen2_vl" in self.model_name:
            image = fetch_image(image,factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,max_ratio=self.max_ratio,)
        return image
