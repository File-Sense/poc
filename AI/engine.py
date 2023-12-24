from os import path
from torch.cuda import is_available
import torchvision.transforms as T  # type: ignore
from PIL import Image
from torch import no_grad, sum, clamp
from transformers import (  # type: ignore
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)
import torch.nn.functional as F
from settings import Settings


class AIEngine(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(AIEngine, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        self.settings = Settings()
        self.device = "cuda" if is_available() else "cpu"
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-caption"),
            local_files_only=True,
        ).to(self.device)
        self.caption_processor = BlipProcessor.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-caption"),
            local_files_only=True,
        )
        self.image_model = AutoModel.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-image"),
            local_files_only=True,
        ).to(self.device)
        self.image_extractor = AutoFeatureExtractor.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-image"),
            local_files_only=True,
        )
        self.text_model = AutoModel.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-text"), local_files_only=True
        ).to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            path.join(self.settings.ROOT_DIR, "AI", "model-text"), local_files_only=True
        )

    def generate_caption(self, image_path: str) -> str:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.caption_processor(
            images=raw_image,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.caption_model.generate(**inputs)
        caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def generate_text_embedding(self, text: list):
        encoded_input = self.text_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        with no_grad():
            model_output = self.text_model(**encoded_input.to(self.device))
        sentence_embedding = self.__mean_pooling(
            model_output, encoded_input["attention_mask"].to(self.device)
        )
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding.tolist()

    def generate_image_embedding(self, image_path: str, image: Image = None):  # type: ignore
        if image is None:
            raw_image = Image.open(image_path).convert("RGB")
        else:
            raw_image = Image.open(image).convert("RGB")
        transformation_chain = T.Compose(
            [
                T.Resize(int((256 / 224) * self.image_extractor.size["height"])),
                T.CenterCrop(self.image_extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(
                    mean=self.image_extractor.image_mean,
                    std=self.image_extractor.image_std,
                ),
            ]
        )

        image_transformed = transformation_chain(raw_image).unsqueeze(0)

        with no_grad():
            image_embedding = (
                self.image_model(image_transformed.to(self.device))
                .last_hidden_state[:, 0]
                .cpu()
            )
            return image_embedding[0].tolist()

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return sum(token_embeddings * input_mask_expanded, 1) / clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
