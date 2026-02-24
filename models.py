from glob import glob
from time  import sleep
import re,os,torch
from prompt import *
import base64,ast
import pandas as pd
from PIL import Image
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq, LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io,base64
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io,cv2
import numpy as np
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import pipeline
from transformers import MllamaForConditionalGeneration, AutoProcessor
import google.api_core.exceptions as google_exceptions

# from deepseek_vl2.utils.io import load_pil_images
# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import base64
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
class AbstractModel():
    def __init__(self,api_key=None):
        self.api_key=api_key
    def prompt_model(self,query,content=None):
    
        pass
    
    def generate_feedback(self,query,question,ref_ans,student_ans,ref_image_path,stud_image_path):
    
        pass

    
    def file_to_data_url(self,file_path: str):
        """
        Helper function to convert a local image file to a data URL.

        Parameters
        ----------
        file_path : str
            The path to the image file

        Return
        -------
        data_url : str
            The data url containing the base64 encoding of the string

        """    
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        _, extension = os.path.splitext(file_path)
        mime_type = f"image/{extension[1:].lower()}"
        
        return f"data:{mime_type};base64,{encoded_string}"

class Dummy(AbstractModel):
    def __str__(self):
        return 'Molmo'
    
    def prompt_model(self,data):
        return "2"
    
class Molmo(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        self.model=AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        # super().__init__(None)
    def __str__(self):
        return 'Molmo'
    
    def prompt_model(self,data=None,extract=True):
        images_process=[Image.open(data['question']['I'])]                                                                                                                                            
        if(not data['answer']['I'] is None):                                                                                                                                                                
            images_process.append(Image.open(data['answer']['I']))                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          

Question : {data['question']['T']}                                                                                                                                                                                                                                                                                                                                                                                       
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""                                                                                                                                                                                                         
        inputs = self.processor.process(                                                                                                                                                                    
            images=images_process,                                                                                                                                                                          
            text=system_prompt+"\n"+user_prompt                                                                                                                                                             
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
            use_cache=False
        )

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response_cleaned=generated_text.replace('\\n',' ')
        if(extract):
            try:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL).group().replace('<Score>','').replace('</Score>','').strip()
            except Exception as e:
                print("Failed to perform Score extraction : ",e)
                return None
            return score
        return response_cleaned

class LLamaVision():
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)     
    
    def __str__(self):
        return 'LLamaVision'
    

    def prompt_model(self,data=None,extract=True):
        images_process=[Image.open(data['question']['I'])]                                                                                                                                             
        if(not data['answer']['I'] is None):                                                                                                                                                                
            images_process.append(Image.open(data['answer']['I']))                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          

Question : {data['question']['T']}                                                                                                                                                                                                                                                                                                                                                                                       
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        user_content=[{"type": "image"} for i in range(len(images_process))]
        user_content.append({"type": "text", "text":system_prompt + "\n" + user_prompt })
        messages = [
            # {"role": "system", "content": [{"type": "text", "text":system_prompt}]},
            {"role": "user", "content": user_content}
            ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images_process,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=2000)
        generated_tokens = output[0]
        generated_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        response_cleaned=generated_text.split('assistant\n\n')[-1]
        response_cleaned=response_cleaned.replace('\\n',' ')
        if(extract):
            try:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL).group().replace('<Score>','').replace('</Score>','').strip()
            except Exception as e:
                print("Failed to perform Score extraction : ",e)
                return None
            return score
        return response_cleaned

class Pixtral(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT,large=False):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        if(large):
           model_name = "mistralai/Pixtral-Large-Instruct-2411"
        else:
           model_name = 'mistralai/Pixtral-12B-2409'
        sampling_params = SamplingParams(max_tokens=8000)
        self.llm = LLM(model=model_name, tokenizer_mode="mistral")
        # super().__init__(None)
    def __str__(self):
        return 'Pixtral'

    def numpy_to_data_url(self,img_array):
        """Convert a NumPy array to a base64 data URL (JPEG)."""
        pil_img = Image.open(img_array.astype("uint8"))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
    
    def prompt_model(self,data=None,extract=True):                                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        content_list = [{"type": "text", "text": self.CL_SCORE_PROMPT+"\n"+user_prompt}]
        content_list.append(self.numpy_to_data_url(data['question']['I']))
        if(not data['answer']['I'] is None):
            content_list.append(self.numpy_to_data_url(data['answer']['I']))

        messages = [
            {
                "role": "user",
                "content": content_list
            },
        ]
        sampling_params = SamplingParams(max_tokens=200)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        response_cleaned=outputs[0].outputs[0].text
        response_cleaned=response_cleaned.replace('\\n',' ')
        if(extract):
            try:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL).group().replace('<Score>','').replace('</Score>','').strip()
            except Exception as e:
                print("Failed to perform Score extraction : ",e,response_cleaned)
                return None
            return score
        return response_cleaned


class Gemini(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT,API_KEY='',model_name='gemini-2.5-flash'):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        self.apikeys=os.environ['GEMINI_API_KEY'].split(',')
        genai.configure(api_key=self.apikeys[0])
        self.model_name=model_name
        # super().__init__(None)

    def __str__(self):
        return "Gemini"

    def numpy_to_base64(self,img_array):
        """Convert a NumPy array to a base64 data URL (JPEG)."""
        pil_img = Image.open(img_array.astype("uint8"))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return base64_str

    def prompt_model(self,data=None,extract=True):
        model = genai.GenerativeModel(self.model_name) # Or 'gemini-pro-vision' for older models
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""   
        content = [
            {
                "role": "user",
                "parts": [

                    {
                        "text": system_prompt+"\n\n"+user_prompt
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",  # or image/png, etc.
                            "data": self.numpy_to_base64(data['question']['I'])
                        }
                    },
                ]
            }
        ]

        if(not data['answer']['I'] is None):
            content[0]['parts'].append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",  # or image/png, etc.
                            "data": self.numpy_to_base64(data['answer']['I'])
                        }
                    },
                )
        try:
            response = model.generate_content(content)
            response_cleaned=response.text
            sleep(8)
        except Exception as e:
            print('Possibly Resource Exhausted : ',e)
            if(len(self.apikeys)>0):
                self.apikeys=self.apikeys[1:]
                genai.configure(api_key=self.apikeys[0])
                response = model.generate_content(content)
                response_cleaned=response.text
            else:
                return '<PLACEHOLDER>'
        response_cleaned=response_cleaned.replace('\\n',' ')
        if(extract):
            try:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL).group().replace('<Score>','').replace('</Score>','').strip()
            except Exception as e:
                print("Failed to perform Score extraction : ",e,response_cleaned)
                return response_cleaned
            return score
        return response_cleaned

    def perform_ocr(self,base64_image): 
        try:
            # Create a GenerativeModel instance
            model = genai.GenerativeModel(self.model_name) # Or 'gemini-pro-vision' for older models

            content = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",  # or image/png, etc.
                                "data": base64_image
                            }
                        },
                        {
                            "text": OCR_PROMPT
                        }
                    ]
                }
            ]

            response = model.generate_content(content)
            return response.text

        except Exception as e:
            print(f"An error occurred: {e}")
        return None

class Gemma(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-3-12b-it",
            device=0 if torch.cuda.is_available() else -1,
        )
        # super().__init__(None)
    def __str__(self):
        return 'Gemma'

    def numpy_to_data_url(self,img_array):
        """Convert a NumPy array to a base64 data URL (JPEG)."""
        pil_img = Image.open(img_array)#.astype("uint8")
        buf = io.BytesIO()
        if(pil_img.mode == "RGBA"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
    def prompt_model(self,data=None,extract=True):                                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        content_list = [{"type": "text", "text": self.CL_SCORE_PROMPT+"\n"+user_prompt}]
        content_list.append(self.numpy_to_data_url(data['question']['I']))
        if(not data['answer']['I'] is None):
            content_list.append(self.numpy_to_data_url(data['answer']['I']))

        messages = [
            {
                "role": "user",
                "content": content_list
            },
        ]
        sampling_params = SamplingParams(max_tokens=200)
        outputs = self.pipe(messages, max_new_tokens=100)
        response_cleaned=outputs[0]['generated_text'][-1]['content']
        response_cleaned=response_cleaned.replace('\\n',' ')
        if(extract):
            if(response_cleaned.strip().isdigit()):
                return response_cleaned
            else:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL)
                if(score is None):
                    temp=response_cleaned.replace('<','').replace('>','')
                    if(temp.isdigit()):
                        return temp
                else:
                    score=score.group().replace('<Score>','').replace('</Score>','').strip()
                    return score
        return response_cleaned

class LLava(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        self.processor=LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model=LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to("cuda:0")
        
    def __str__(self):
        return 'LLava'

    def prompt_model(self,data=None,extract=True):                                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        
        content_list = [{"type": "image"}]
        images = [Image.open(data['question']['I'])]
        if(not data['answer']['I'] is None):
            images.append(Image.open(data['answer']['I']))
            content_list.append({"type": "image"})
        content_list.append({"type": "text", "text": self.CL_SCORE_PROMPT+"\n"+user_prompt})
        messages = [
            {
                "role": "user",
                "content": content_list
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")
        output = self.model.generate(**inputs, max_new_tokens=200)
        response_cleaned=self.processor.decode(output[0], skip_special_tokens=True)
        response_cleaned=response_cleaned.split('[/INST]')[-1].replace('\\n',' ')
        if(extract):
            if(response_cleaned.strip().isdigit()):
                return response_cleaned
            else:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL)
                if(score is None):
                    temp=response_cleaned.replace('<','').replace('>','')
                    if(temp.isdigit()):
                        return temp
                else:
                    score=score.group().replace('<Score>','').replace('</Score>','').strip()
                    return score
        return response_cleaned


class Granite(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        model_path = "ibm-granite/granite-vision-3.3-2b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

    def __str__(self):
        return 'Granite'

    def numpy_to_data_url(self,img_array):
        """Convert a NumPy array to a base64 data URL (JPEG)."""
        pil_img = Image.open(img_array)
        buf = io.BytesIO()
        if(pil_img.mode == "RGBA"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image", "url": f"data:image/jpeg;base64,{base64_str}"}

    def prompt_model(self,data=None,extract=True):                                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        content_list = [{"type": "text", "text": user_prompt}]
        content_list.append(self.numpy_to_data_url(data['question']['I']))
        if(not data['answer']['I'] is None):
            content_list.append(self.numpy_to_data_url(data['answer']['I']))

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.CL_SCORE_PROMPT}]
            },
            {
                "role": "user",
                "content": content_list
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)


        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100)
        response_cleaned = self.processor.decode(output[0], skip_special_tokens=True)
        response_cleaned=response_cleaned.split('\n\n<|assistant|>\n')[-1].replace('\\n',' ')
        if(extract):
            if(response_cleaned.strip().isdigit()):
                return response_cleaned
            else:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL)
                if(score is None):
                    temp=response_cleaned.replace('<','').replace('>','')
                    if(temp.isdigit()):
                        return temp
                else:
                    score=score.group().replace('<Score>','').replace('</Score>','').strip()
                    return score
        return response_cleaned

class Qwen(AbstractModel):
    def __init__(self,CL_SCORE_PROMPT):
        self.CL_SCORE_PROMPT=CL_SCORE_PROMPT
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def __str__(self):
        return 'Qwen'

    def numpy_to_data_url(self,img_array):
        """Convert a NumPy array to a base64 data URL (JPEG)."""
        pil_img = Image.open(img_array)
        if(pil_img.mode == "RGBA"):
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image", "image_url": f"data:image/jpeg;base64,{base64_str}"}

    def prompt_model(self,data=None,extract=True):                                                                                                                                                     
        system_prompt=self.CL_SCORE_PROMPT.format(cl_desc=data['concept_link_score'])                                                                                                                                                                       
        user_prompt=f"""                                                                                                                                                                                    
Here is the Question, Concept Link and Student Answer.                                                                                                                                                      
Note that in the images provided the first image belongs to the question{'' if data['answer']['I'] is None else ' and the second image if present is part of the student answer'}.                          
                                                                                                                                                                                                            
Question : {data['question']['T']}                                                                                                                                                                          
Student Answer : {data['answer']['T'] if data['answer']['T'] else None}                                                                                                                                     
Concept Link : {data['concept_link']}                                                                                                                                                                       
"""     
        content_list = []
        content_list.append(self.numpy_to_data_url(data['question']['I']))
        if(not data['answer']['I'] is None):
            content_list.append(self.numpy_to_data_url(data['answer']['I']))
        content_list.append({"type": "text", "text": user_prompt})
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": content_list
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=2000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response_cleaned = output_text[0]
        response_cleaned=response_cleaned.split('\n\n<|assistant|>\n')[-1].replace('\\n',' ')
        if(extract):
            if(response_cleaned.strip().isdigit()):
                return response_cleaned
            else:
                score=re.search('<Score>.*</Score>',response_cleaned,flags=re.DOTALL)
                if(score is None):
                    temp=response_cleaned.replace('<','').replace('>','')
                    if(temp.isdigit()):
                        return temp
                else:
                    score=score.group().replace('<Score>','').replace('</Score>','').strip()
                    return score
        return response_cleaned