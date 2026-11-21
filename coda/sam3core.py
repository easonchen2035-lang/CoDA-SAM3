"""Frozen SAM3 (HF transformers `Sam3Model`) split into reusable stages.

Stages (all under torch.no_grad, weights frozen):
  encode_image(img)              -> ImageEnc (vision outputs + dense features), computed once per image
  encode_text(phrase)            -> (text_tokens [1,L,256], text_mask [1,L])  (cached per phrase)
  geometry_tokens(enc, boxes)    -> exemplar tokens from SAM3's own geometry encoder, using the
                                    features of the image `enc` that the boxes live in.
  decode(enc_q, prompt_tokens)   -> raw detections on the query image (scores, boxes, mask logits)

`decode` replicates `Sam3Model.forward` from the point where text and geometry prompt tokens are
concatenated (transformers 5.x, modeling_sam3.py). Native SAM3 usage computes geometry tokens on the
*same* image that is decoded. CoDA-SAM3 additionally supports *cross-image* exemplar tokens: geometry
tokens computed on a support image (ROI-pooled support features + cross-attention to support features)
and then appended to the prompt sequence for the query image. This is a modification of how SAM3 is
used (out-of-distribution for its training); it is documented in METHOD_IMPLEMENTATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import box_cxcywh_to_xyxy, inverse_sigmoid

MODEL_ID = "facebook/sam3"


@dataclass
class ImageEnc:
    vision: object            # Sam3VisionEncoderOutput (fpn_hidden_states, fpn_position_encoding, last_hidden_state)
    orig_hw: tuple            # (H, W) of the original image
    feat: torch.Tensor        # [C, 72, 72] L2-normalised dense feature used for prototypes/response (float32)
    gfeat: torch.Tensor       # [C] L2-normalised global descriptor (GAP of feat)


class FrozenSAM3:
    def __init__(self, device="cuda", dtype=torch.bfloat16, feat_source="vit"):
        self.device, self.dtype = device, dtype
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        self.model = Sam3Model.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert feat_source in ("vit", "fpn72")
        self.feat_source = feat_source
        self._text_cache = {}

    # ------------------------------------------------------------------ encoders
    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> ImageEnc:
        img = img.convert("RGB")
        inp = self.processor(images=img, return_tensors="pt")
        pv = inp["pixel_values"].to(self.device, self.dtype)
        vis = self.model.get_vision_features(pixel_values=pv)
        if self.feat_source == "vit":
            # ViT backbone output (1024-d, 72x72 grid for 1008px input), before the FPN neck.
            h = vis.last_hidden_state
            if h.dim() == 3:  # [B, N, C]
                n = h.shape[1]; s = int(round(n ** 0.5))
                h = h[0].transpose(0, 1).reshape(-1, s, s)
            else:             # [B, H, W, C] or [B, C, H, W]
                h = h[0]
                if h.shape[-1] > h.shape[0]:
                    h = h.permute(2, 0, 1)
            feat = h
        else:
            feat = vis.fpn_hidden_states[2][0]  # 256-d, 72x72: the level used by the geometry encoder
        feat = F.normalize(feat.float(), dim=0)
        g = F.normalize(feat.mean((1, 2)), dim=0)
        return ImageEnc(vision=vis, orig_hw=(img.height, img.width), feat=feat, gfeat=g)

    @torch.no_grad()
    def encode_text(self, phrase: str):
        if phrase not in self._text_cache:
            t = self.processor(text=phrase, return_tensors="pt").to(self.device)
            out = self.model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
            self._text_cache[phrase] = (out.pooler_output.to(self.dtype), t["attention_mask"].bool())
        return self._text_cache[phrase]

    @torch.no_grad()
    def geometry_tokens(self, enc: ImageEnc, boxes_xyxy, labels=None, coord_mode="native", drop_cls=False):
        """SAM3 geometry-encoder tokens for boxes (pixel xyxy in `enc`'s original image).

        coord_mode:
          native : exactly SAM3's geometry encoder (direct coord projection + ROI-pooled feature
                   + sine pos-enc of the box centre + label embedding). Correct for in-image prompts.
          nopos  : cross-image use; drops the two coordinate terms (direct projection and centre/size
                   pos-enc) so the token carries only ROI-pooled appearance + label (+ the encoder's
                   cross-attention over the source image). Avoids copying the support box *location*.
          center : cross-image use; coordinate terms are computed for a box with the same w,h but
                   re-centred at (0.5,0.5): keeps size info, removes location.
        Returns tokens [1, n(+1 cls), 256] and mask [1, n(+1)]."""
        g = self.model.geometry_encoder
        H, W = enc.orig_hw
        b = torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32), device=self.device).reshape(-1, 4)
        b = (b / torch.tensor([W, H, W, H], device=self.device, dtype=torch.float32)).clamp(0, 1)
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                              (b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])], -1)[None].to(self.dtype)
        n = cxcywh.shape[1]
        lab = torch.ones(1, n, dtype=torch.long, device=self.device) if labels is None \
            else torch.as_tensor(labels, device=self.device).long()[None]
        msk = torch.ones_like(lab, dtype=torch.bool)
        fpn = enc.vision.fpn_hidden_states[:-1]
        pos = enc.vision.fpn_position_encoding[:-1]
        if coord_mode == "native":
            out = g(box_embeddings=cxcywh, box_mask=msk, box_labels=lab, img_feats=fpn, img_pos_embeds=pos)
            tok, m = out.last_hidden_state, out.attention_mask
        else:
            tok, m = self._geometry_custom(g, cxcywh, msk, lab, fpn, pos, coord_mode)
        if drop_cls:
            tok, m = tok[:, :-1], m[:, :-1]
        return tok, m

    def _geometry_custom(self, g, cxcywh, msk, lab, fpn, pos, coord_mode):
        """Re-implementation of Sam3GeometryEncoder.forward with modified coordinate terms
        (see transformers modeling_sam3.Sam3GeometryEncoder; layer/norm weights are SAM3's own)."""
        import torchvision
        from transformers.masking_utils import create_bidirectional_mask
        from transformers.models.sam3.modeling_sam3 import concat_padded_sequences
        vf = fpn[-1]
        vpos = pos[-1]
        vflat, pflat = vf.flatten(2).transpose(1, 2), vpos.flatten(2).transpose(1, 2)
        nf = g.vision_layer_norm(vf.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h, w = nf.shape[-2:]
        xyxy = box_cxcywh_to_xyxy(cxcywh) * torch.tensor([w, h, w, h], dtype=cxcywh.dtype, device=cxcywh.device)
        dt = torch.float16 if nf.dtype == torch.bfloat16 else nf.dtype
        roi = torchvision.ops.roi_align(nf.to(dt), xyxy.to(dt).unbind(0), g.roi_size).to(nf.dtype)
        emb = g.boxes_pool_project(roi).view(1, cxcywh.shape[1], g.hidden_size)
        if coord_mode == "center":
            c = cxcywh.clone(); c[..., 0] = 0.5; c[..., 1] = 0.5
            emb = emb + g.boxes_direct_project(c)
            cx, cy, bw, bh = c.unbind(-1)
            pe = g._encode_box_coordinates(cx.flatten(), cy.flatten(), bw.flatten(), bh.flatten())
            emb = emb + g.boxes_pos_enc_project(pe.view(1, c.shape[1], -1))
        elif coord_mode != "nopos":
            raise ValueError(coord_mode)
        emb = emb + g.label_embed(lab)
        cls = g.cls_embed.weight.view(1, 1, g.hidden_size)
        emb, m = concat_padded_sequences(emb, msk, cls, torch.ones(1, 1, dtype=msk.dtype, device=msk.device))
        emb = g.prompt_layer_norm(g.final_proj(emb))
        am = create_bidirectional_mask(config=g.config, inputs_embeds=emb, attention_mask=m)
        for layer in g.layers:
            emb = layer(prompt_feats=emb, vision_feats=vflat, vision_pos_encoding=pflat, prompt_mask=am)
        return g.output_layer_norm(emb), m

    # ------------------------------------------------------------------ decoder
    @torch.no_grad()
    def decode(self, enc: ImageEnc, prompt_tokens: torch.Tensor, prompt_mask: torch.Tensor):
        """Run SAM3 DETR encoder/decoder + mask decoder on query `enc` with an arbitrary prompt sequence.
        Returns dict of float32 tensors: scores [Q] (sigmoid(logit) * sigmoid(presence)), boxes_xyxy_norm [Q,4],
        mask_logits [Q,288,288], presence scalar."""
        m = self.model
        fpn = enc.vision.fpn_hidden_states[:-1]
        pos = enc.vision.fpn_position_encoding[:-1]
        enc_out = m.detr_encoder(vision_features=[fpn[-1]], text_features=prompt_tokens,
                                 vision_pos_embeds=[pos[-1]], text_mask=prompt_mask)
        dec = m.detr_decoder(vision_features=enc_out.last_hidden_state, text_features=enc_out.text_features,
                             vision_pos_encoding=enc_out.pos_embeds_flattened, text_mask=prompt_mask,
                             spatial_shapes=enc_out.spatial_shapes)
        hs = dec.intermediate_hidden_states
        offs = m.detr_decoder.box_head(hs[-1:])
        boxes = box_cxcywh_to_xyxy((inverse_sigmoid(dec.reference_boxes[-1:]) + offs).sigmoid())[-1]
        logits = m.dot_product_scoring(decoder_hidden_states=hs[-1:], text_features=enc_out.text_features,
                                       text_mask=prompt_mask).squeeze(-1)[-1]
        presence = dec.presence_logits[-1]
        mo = m.mask_decoder(decoder_queries=hs[-1], backbone_features=list(fpn),
                            encoder_hidden_states=enc_out.last_hidden_state,
                            prompt_features=prompt_tokens, prompt_mask=prompt_mask)
        pres = presence.float().sigmoid().reshape(-1)[0]
        scores = logits.float().sigmoid()[0] * pres
        return {"scores": scores, "boxes": boxes[0].float().clamp(0, 1),
                "mask_logits": mo.pred_masks[0].float(), "presence": float(pres)}

    @torch.no_grad()
    def text_prompt(self, phrase):
        return self.encode_text(phrase)


def cat_prompts(parts):
    """Concatenate (tokens, mask) pairs along the sequence dim."""
    parts = [p for p in parts if p is not None and p[0].shape[1] > 0]
    return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
