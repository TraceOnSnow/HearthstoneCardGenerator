from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder(nn.Module):
    """CLIP-based encoder for Hearthstone card images and text descriptions.

    Wraps a pretrained CLIP model and exposes separate image and text
    encoding methods as well as a projection head to map embeddings into
    a shared latent space used by the GNN model.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 256,
        freeze_clip: bool = True,
    ) -> None:
        """Initialise the CLIP encoder.

        Args:
            model_name: HuggingFace model identifier for the CLIP checkpoint.
            projection_dim: Output dimensionality of the projection head.
            freeze_clip: If True, CLIP weights are frozen during training.
        """
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        clip_dim = self.clip.config.projection_dim

        self.image_projection = nn.Sequential(
            nn.Linear(clip_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(clip_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of card images.

        Args:
            pixel_values: Pre-processed image tensor of shape (B, C, H, W).

        Returns:
            Projected image embeddings of shape (B, projection_dim).
        """
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.image_projection(image_features)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of card text descriptions.

        Args:
            input_ids: Tokenised text tensor of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Projected text embeddings of shape (B, projection_dim).
        """
        text_features = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.text_projection(text_features)

    def forward(
        self,
        pixel_values: torch.Tensor | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Encode both image and text for a batch of cards.

        Args:
            pixel_values: Pre-processed image tensor of shape (B, C, H, W),
                or ``None`` when images are unavailable.
            input_ids: Tokenised text tensor of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Tuple of (image_embeddings, text_embeddings).  *image_embeddings*
            is ``None`` when *pixel_values* is ``None``; otherwise it has
            shape (B, projection_dim).  *text_embeddings* always has shape
            (B, projection_dim).
        """
        image_emb = self.encode_image(pixel_values) if pixel_values is not None else None
        text_emb = self.encode_text(input_ids, attention_mask)
        return image_emb, text_emb

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def preprocess(self, images=None, texts=None) -> dict:
        """Run the CLIP processor on raw images and/or texts.

        Args:
            images: PIL images or list of PIL images.
            texts: String or list of strings.

        Returns:
            Dictionary of tensors ready to be passed to :meth:`forward`.
        """
        return self.processor(images=images, text=texts, return_tensors="pt", padding=True)
