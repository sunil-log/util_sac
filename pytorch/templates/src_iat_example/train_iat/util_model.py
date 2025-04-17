import torch
import torch.nn as nn

from util_sac.pytorch.print_array import print_array_info
from util_sac.pytorch.model.positional_encoding import positional_encoding
from util_sac.pytorch.model.projectors.stacked_projectors import StackedProjector
from util_sac.pytorch.model.AttnPooling import MultiHeadAttnPoolingWithMask




class IAT_model(nn.Module):
	def __init__(self, args):
		super().__init__()

		arm = args["model"]  # args["model"]를 arm으로 할당
		self.uses = args["use"]  # args["uses"]를 self.uses로 할당


		# projection layer
		self.proj_dmgs = StackedProjector(
			in_dim=8,
			hidden_dim=arm["emb_dim"],
			out_dim=arm["emb_dim"],
			num_stacks=arm["proj_stack"],
			p_dropout=arm["dropout"],
		)

		self.proj_wat_select = StackedProjector(
			in_dim=6,
			hidden_dim=arm["emb_dim"],
			out_dim=arm["emb_dim"],
			num_stacks=arm["proj_stack"],
			p_dropout=arm["dropout"],
		)



		# attention pooling for iat
		iat_dim = 8 if args["iat_use_name"] else 4
		self.att_iat_1 = MultiHeadAttnPoolingWithMask(
			input_dim=iat_dim,
			hidden_dim=arm["q_dim"]*arm["n_head"],
			num_heads=arm["n_head"],
			output_dim=arm["emb_dim"],
			dropout_p=arm["dropout"],
		)
		self.att_iat_2 = MultiHeadAttnPoolingWithMask(
			input_dim=iat_dim,
			hidden_dim=arm["q_dim"]*arm["n_head"],
			num_heads=arm["n_head"],
			output_dim=arm["emb_dim"],
			dropout_p=arm["dropout"],
		)


		# wat attention pooling
		self.att_wat_1 = MultiHeadAttnPoolingWithMask(
			input_dim=6,
			hidden_dim=arm["q_dim"]*arm["n_head"],
			num_heads=arm["n_head"],
			output_dim=arm["emb_dim"],
			dropout_p=arm["dropout"],
		)
		self.att_wat_2 = MultiHeadAttnPoolingWithMask(
			input_dim=6,
			hidden_dim=arm["q_dim"]*arm["n_head"],
			num_heads=arm["n_head"],
			output_dim=arm["emb_dim"],
			dropout_p=arm["dropout"],
		)



		# 학습 대상이 아닌 Positional Encoding을 생성
		pe_wat = positional_encoding(45, 6)     # (1, 45, 6)
		self.register_buffer("pe_wat", pe_wat)      # register_buffer를 통해 모델과 함께 이동하되 학습은 되지 않도록 등록


		self.w_iat_1 = nn.Parameter(torch.tensor(1.0))
		self.w_iat_2 = nn.Parameter(torch.tensor(1.0))
		self.w_wat_1 = nn.Parameter(torch.tensor(1.0))
		self.w_wat_2 = nn.Parameter(torch.tensor(1.0))
		self.w_wat_s = nn.Parameter(torch.tensor(1.0))


		# logit 값을 계산하기 위한 projection layer
		self.proj_logit = nn.Linear(arm["emb_dim"], 2)




	def forward(self, x):

		"""

		:param x:
		:return:
		"""



		"""
		path       PyTorch Tensor       (16,)                       128.00 B torch.int64
		y          PyTorch Tensor       (16,)                       128.00 B torch.int64
		demographics PyTorch Tensor       (16, 8)                     512.00 B torch.float32
		iat_valid_mask PyTorch Tensor       (16, 2, 40)                 10.00 KB torch.int64
		iat_emb    PyTorch Tensor       (16, 2, 40, 4)              20.00 KB torch.float32
		wat_emb    PyTorch Tensor       (16, 2, 45, 6)              33.75 KB torch.float32
		wat_select PyTorch Tensor       (16, 6)                     384.00 B torch.float32
		"""


		# projection
		z_dmgs = self.proj_dmgs(x['demographics'])          # (16, 8) -> (16, 64)
		z_wat_select = self.proj_wat_select(x['wat_select'])

		"""
		attention for iat
		"""
		iat_emb_1 = x['iat_emb'][:, 0, :, :]
		iat_emb_2 = x['iat_emb'][:, 1, :, :]
		iat_valid_mask_1 = x['iat_valid_mask'][:, 0, :]
		iat_valid_mask_2 = x['iat_valid_mask'][:, 1, :]
		z_iat_1, _ = self.att_iat_1(iat_emb_1, iat_valid_mask_1)        # (16, 40, 4) -> (16, 64)
		z_iat_2, _ = self.att_iat_2(iat_emb_2, iat_valid_mask_2)        # (16, 40, 4) -> (16, 64)


		"""
		attention for wat
		"""
		wat_emb_1 = x['wat_emb'][:, 0, :, :]
		wat_emb_2 = x['wat_emb'][:, 1, :, :]
		z_wat_emb_1, _ = self.att_wat_1(wat_emb_1)        # (16, 45, 6) -> (16, 64)
		z_wat_emb_2, _ = self.att_wat_2(wat_emb_2)        # (16, 45, 6) -> (16, 64)



		# z
		z = z_dmgs
		if self.uses["wat_s"]:
			z += self.w_wat_s * z_wat_select
		if self.uses["iat"]:
			z += self.w_iat_1 * z_iat_1 + self.w_iat_2 * z_iat_2
		if self.uses["wat"]:
			z += self.w_wat_1 * z_wat_emb_1 + self.w_wat_2 * z_wat_emb_2


		# logits
		logits = self.proj_logit(z)


		return logits
