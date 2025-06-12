
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
import base64
from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

import ipdb
import json
_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'


with open('/scratch/ramya/C-TPT/clip/StanfordCars_2_V.json', 'r') as file:
    json_data = file.read()

# Parse the JSON data
data = json.loads(json_data)

# Convert the values to lists
attributes = {key: value.split() for key, value in data.items()}


class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    # prompts.append(prompt_prefix + " " + name + " " + attribute + ".")
                    prompts.append(prompt_prefix + " " + attribute + " " + name + ".")
            # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    prompts.append(prompt_prefix + " " + attribute + " " + cls_token + ".")
                    # prompts.append(prompt_prefix + " " + cls_token + " " + attribute + ".")
            # prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self, classnames, arch, image):
        from openai import OpenAI
        from PIL import Image
        import io
        import base64
        from torchvision import transforms


        client = OpenAI()

        # Function to encode the image
        def encode_image(image):
            return base64.b64encode(image).decode('utf-8')
        
        def tensor_to_base64(image_tensor):
            # Convert the tensor to a PIL image (assuming the tensor is in the shape [C, H, W])
            pil_image = transforms.ToPILImage()(image_tensor)
            
            # Save the image to a bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")  # You can choose other formats like "JPEG"
            buffer.seek(0)
            
            # Encode the bytes buffer to a base64 string
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return image_base64


        
        # Getting the base64 string
        base64_image = tensor_to_base64(image.squeeze(0))
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Please take a look at this image file and provide only 1 descriptive English word for the this image.",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        )

        image_attribute = response.choices[0].message.content.replace(".", "")
        




        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # Modification
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    prompts.append(self.prompt_prefix + " " + attribute + " " + image_attribute + " " + name + ".")
                    # prompts.append(self.prompt_prefix + " " + name + " " + attribute + ".")
            # prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    prompts.append(self.prompt_prefix + " " + attribute + " " + image_attribute + " " + cls_token + ".")
                    # prompts.append(self.prompt_prefix + " " + cls_token + " " + attribute + ".")
            # prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames


        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)
    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            # Modification
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    prompts.append(self.prompt_prefix + " " + attribute + " " + name + ".")
                    # prompts.append(self.prompt_prefix + " " + name + " " + attribute + ".")
            # prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = []
            for name in classnames:
                lst = attributes[name]
                for attribute in lst[:2]:
                    prompts.append(self.prompt_prefix + " " + attribute + " " + cls_token + ".")
                    # prompts.append(self.prompt_prefix + " " + cls_token + " " + attribute + ".")
            # prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
            
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.token_prefix.shape[0], -1, -1)
            
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)

        # self.image_encoder = self.image_encoder.to(device)
        # self.text_encoder = self.text_encoder.to(device)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     self.image_encoder = torch.nn.DataParallel(self.image_encoder, device_ids=list(range(torch.cuda.device_count())))
        #     self.text_encoder = torch.nn.DataParallel(self.text_encoder, device_ids=list(range(torch.cuda.device_count())))

        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        
    # @property
    # def dtype(self):
    #     return self.image_encoder.conv1.weight.dtype
    @property
    def dtype(self):
        # Handle the case when image_encoder is wrapped in DataParallel
        if isinstance(self.image_encoder, torch.nn.DataParallel):
            return self.image_encoder.module.conv1.weight.dtype
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self, classnames, arch, image):
        return self.prompt_learner.reset(classnames, arch, image)

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features() # (1000, 512)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        #[c-tpt] --------------------------------------------
        if self.l2_norm_cal:
            attribute_features = text_features.reshape(-1, 3, text_features.shape[1]) # 10
            attribute_mean = attribute_features.mean(dim=1) # Centroid of each class
            class_mean = attribute_mean.mean(0) # (512) # Centroid of whole feature space
            feature_class_distance = attribute_mean - class_mean
            l2_norm = torch.linalg.norm(feature_class_distance, dim=-1)
            l2_norm_mean = l2_norm.mean()

            
            self.l2_norm_mean = l2_norm_mean.item()
            self.l2_norm_mean_training = l2_norm_mean

            attribute_mean = attribute_mean.reshape(-1, 1, attribute_mean.shape[1]) # (100, 1, 512)
            feature_distance = attribute_features - attribute_mean 
            l2_norm = torch.linalg.norm(feature_distance, dim=-1)
            
            l2_norm_attribute_mean = l2_norm.mean(dim=-1)
            self.l2_norm_attribute_mean_training = l2_norm_attribute_mean
        
            # prompt_mean = text_features.mean(0)
            # feature_distance = text_features - prompt_mean
            # l2_norm = torch.linalg.norm(feature_distance, dim=-1)
            # l2_norm_mean = l2_norm.mean()
            
            # #for saving to csv file
            # self.l2_norm_mean = l2_norm_mean.item()
            
            # #for training
            # self.l2_norm_mean_training = l2_norm_mean



            # self.model.l2_norm_attribute_mean_training
        
        #-----------------------------------------------------

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t() # (1, 1000) or (64,1000)
        logits = logits.reshape(logits.shape[0],-1 ,3) # (1, 100, 10) or (64, 100, 10) 
        logits = torch.exp(logits)
        logits = torch.sum(logits, dim=-1, keepdim=True)
        sum = torch.sum(logits, dim=1, keepdim=True)
        logits /= sum
        logits = logits.squeeze(-1)
        # print("logits.shape", logits)
        return logits

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

