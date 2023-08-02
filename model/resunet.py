# coding = utf-8
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm

from model.residual_block import get_block
from model.Img_Encoder import ImageEncoder
from model.attention_fusion import AttentionFusion

import torch.nn as nn
import math

class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # IMG_CHANNELS = [None, 64, 128, 256, 512]
  IMG_CHANNELS = [None, 0, 0, 0, 0]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3,
               config=None):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE

    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    IMG_CHANNELS = self.IMG_CHANNELS

    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)
    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)
    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)
    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)
    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    # fusion attention
    self.attention_fusion = AttentionFusion(
        dim=128,  # the image channels
        depth=1,  # depth of net (self-attention - Processing的数量)
        latent_dim=CHANNELS[4],  # the PC channels
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=int(CHANNELS[4]/2),  # number of dimensions per cross attention head
        latent_dim_head=int(CHANNELS[4]/2),  # number of dimensions per latent self attention head
    )

    self.attention_fusion_ptp = AttentionFusion(
        dim=CHANNELS[4],  # the image channels
        depth=1,  # depth of net (self-attention - Processing的数量)
        latent_dim=CHANNELS[4],  # the PC channels
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=8,  # number of heads for latent self attention, 8
        cross_dim_head=int(CHANNELS[4]/2),  # number of dimensions per cross attention head
        latent_dim_head=int(CHANNELS[4]/2),  # number of dimensions per latent self attention head
    )

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4] + IMG_CHANNELS[1],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3] + IMG_CHANNELS[2],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2] + IMG_CHANNELS[3],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

    # image_Encoder
    self.img_encoder = ImageEncoder()

  def forward(self, stensor_src, stensor_tgt, src_image, tgt_image):

    # I1,I2,I3,I_global = self.img_encoder(image)
    src_image = self.img_encoder(src_image)
    tgt_image = self.img_encoder(tgt_image)

    # Encode src
    src_s1 = self.conv1(stensor_src)
    src_s1 = self.norm1(src_s1)
    src_s1 = self.block1(src_s1)
    src = MEF.relu(src_s1)

    src_s2 = self.conv2(src)
    src_s2 = self.norm2(src_s2)
    src_s2 = self.block2(src_s2)
    src = MEF.relu(src_s2)

    src_s4 = self.conv3(src)
    src_s4 = self.norm3(src_s4)
    src_s4 = self.block3(src_s4)
    src = MEF.relu(src_s4)

    src_s8 = self.conv4(src)
    src_s8 = self.norm4(src_s8)
    src_s8 = self.block4(src_s8)
    src = MEF.relu(src_s8)

    # Encode tgt
    tgt_s1 = self.conv1(stensor_tgt)
    tgt_s1 = self.norm1(tgt_s1)
    tgt_s1 = self.block1(tgt_s1)
    tgt = MEF.relu(tgt_s1)

    tgt_s2 = self.conv2(tgt)
    tgt_s2 = self.norm2(tgt_s2)
    tgt_s2 = self.block2(tgt_s2)
    tgt = MEF.relu(tgt_s2)

    tgt_s4 = self.conv3(tgt)
    tgt_s4 = self.norm3(tgt_s4)
    tgt_s4 = self.block3(tgt_s4)
    tgt = MEF.relu(tgt_s4)

    tgt_s8 = self.conv4(tgt)
    tgt_s8 = self.norm4(tgt_s8)
    tgt_s8 = self.block4(tgt_s8)
    tgt = MEF.relu(tgt_s8)
    # print(out_s.coordinate_manager)
    # print(out_s4.coordinate_manager)

    # fusion-attention
    # src._F = self.transformer(images=src_image, F=src.F,xyz=src.C)
    out_sfs_feat0 = self.transformer(images=src_image, F=src.F,xyz=src.C)
    out_sft_feat0 = self.transformer(images=src_image, F=tgt.F,xyz=tgt.C)
    src._F = self.transformer_ptp(F0=out_sfs_feat0, xyz0=src.C, F1=out_sft_feat0)
    # print(out_s_feat.shape)
    # print(out_s.C.shape)

    out_sfs_feat1 = self.transformer(images=tgt_image, F=src.F, xyz=src.C)
    out_sft_feat1 = self.transformer(images=tgt_image, F=tgt.F, xyz=tgt.C)
    tgt._F = self.transformer_ptp(F0=out_sft_feat1, xyz0=tgt.C, F1=out_sfs_feat1)
    # print(out_t_feat.shape)
    # print(out_t.C.shape)
    
    # Decode src
    src = self.conv4_tr(src)
    src = self.norm4_tr(src)
    src = self.block4_tr(src)
    src_s4_tr = MEF.relu(src)

    # print(out_s4_tr.coordinate_manager)
    src = ME.cat(src_s4_tr, src_s4)
    del src_s4_tr
    del src_s4

    src = self.conv3_tr(src)
    src = self.norm3_tr(src)
    src = self.block3_tr(src)
    src_s2_tr = MEF.relu(src)

    src = ME.cat(src_s2_tr, src_s2)
    del src_s2_tr
    del src_s2

    src = self.conv2_tr(src)
    src = self.norm2_tr(src)
    src = self.block2_tr(src)
    src_s1_tr = MEF.relu(src)

    src = ME.cat(src_s1_tr, src_s1)
    del src_s1_tr
    del src_s1

    src= self.conv1_tr(src)
    src = MEF.relu(src)
    src = self.final(src)

    # Decode tst
    tgt = self.conv4_tr(tgt)
    tgt = self.norm4_tr(tgt)
    tgt = self.block4_tr(tgt)
    tgt_s4_tr = MEF.relu(tgt)

   
    tgt = ME.cat(tgt_s4_tr, tgt_s4)
    del tgt_s4_tr
    del tgt_s4

    tgt = self.conv3_tr(tgt)
    tgt = self.norm3_tr(tgt)
    tgt = self.block3_tr(tgt)
    tgt_s2_tr = MEF.relu(tgt)

    tgt = ME.cat(tgt_s2_tr, tgt_s2)
    del tgt_s2_tr
    del tgt_s2

    tgt = self.conv2_tr(tgt)
    tgt = self.norm2_tr(tgt)
    tgt = self.block2_tr(tgt)
    tgt_s1_tr = MEF.relu(tgt)

    tgt = ME.cat(tgt_s1_tr, tgt_s1)
    del tgt_s1_tr
    del tgt_s1

    tgt = self.conv1_tr(tgt)
    tgt = MEF.relu(tgt)
    tgt = self.final(tgt)

    if self.normalize_feature:
      return ME.SparseTensor(
          src.F / torch.norm(src.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=src.coordinate_map_key,
          coordinate_manager=src.coordinate_manager
      ), ME.SparseTensor(
          tgt.F / torch.norm(tgt.F, p=2, dim=1, keepdim=True),
          coordinate_map_key=tgt.coordinate_map_key,
          coordinate_manager=tgt.coordinate_manager
      )
    else:
      return src, tgt

  def transformer(self,images,F,xyz):

      # batch ----- batch
      lengths = []
      max_batch = torch.max(xyz[:, 0])
      for i in range(max_batch + 1):
          length = torch.sum(xyz[:, 0] == i)
          lengths.append(length)
      # batch ----- batch

      ps = []
      start = 0
      end = 0
      for length,image in zip(lengths,images):

          # pc ------- pc
          end += length
          P_att = torch.unsqueeze(F[start:end, :], dim=0)  # [B,M,C]
          # pc ------- pc

          # image ------- image
          image = torch.unsqueeze(image,dim=0)
          B,C,H,W = image.shape
          image = image.view(B,C,H*W)
          image = image.permute(0,2,1) # [B,H*W,C]
          # image ------- image

          # fusion attention
          P_att = self.attention_fusion(image,queries_encoder = P_att)
          P_att = torch.squeeze(P_att)
          start += length
          ps.append(P_att)
          # fusion attention

      F = torch.cat(ps, dim=0)

      return F

  def transformer_ptp(self, F0, xyz0, F1):

      # batch ----- batch
      lengths = []
      max_batch = torch.max(xyz0[:, 0])
      for i in range(max_batch + 1):
          length = torch.sum(xyz0[:, 0] == i)
          lengths.append(length)
      # batch ----- batch

      ps = []
      start = 0
      end = 0
      for length in lengths:

          # pc ------- pc
          end += length
          P0_att = torch.unsqueeze(F0[start:end, :], dim=0)  # [B,M,C]
          P1_att = torch.unsqueeze(F1[start:end, :], dim=0)
          # pc ------- pc


          # fusion attention
          P_att = self.attention_fusion_ptp(P1_att,queries_encoder = P0_att)
          P_att = torch.squeeze(P_att)
          start += length
          ps.append(P_att)
          # fusion attention

      F = torch.cat(ps, dim=0)

      return F

class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'
