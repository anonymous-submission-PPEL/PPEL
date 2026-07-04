import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.omics_encoder import SNN_Block
from utils.loss_func import NLLSurvLoss


BAG_SIZE = 512


class PathwayPrognosticEvidenceLearning(nn.Module):
    """
    Pathway-conditioned Prognostic Evidence Learning (PPEL).

    The module uses pathway tokens to localize pathology evidence in WSI
    patches, recovers pathway-level prognostic evidence from pathology, and
    adaptively fuses recovered and observed pathway evidence when genomics are
    available. When genomics are missing, learned pathway queries retrieve WSI
    evidence directly.
    """
    def __init__(
            self,
            wsi_dim,
            pathway_dim,
            num_pathways,
            topk_ratio=0.5,
            comp_margin=0.75,
            use_pel=True,
            use_mel=True,
            use_cgf=True,
    ):
        super().__init__()
        self.pathway_dim = pathway_dim
        self.num_pathways = num_pathways
        self.topk_ratio = topk_ratio
        self.comp_margin = comp_margin
        self.use_pel = bool(use_pel)
        self.use_mel = bool(use_mel)
        self.use_cgf = bool(use_cgf)

        self.wsi_proj = nn.Sequential(
            nn.Linear(wsi_dim, pathway_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(pathway_dim, pathway_dim),
        )
        self.pathway_queries = nn.Parameter(torch.randn(1, num_pathways, pathway_dim) * 0.02)

        self.q_proj = nn.Linear(pathway_dim, pathway_dim)
        self.k_proj = nn.Linear(pathway_dim, pathway_dim)
        self.v_proj = nn.Linear(pathway_dim, pathway_dim)

        self.recover = nn.Sequential(
            nn.LayerNorm(pathway_dim),
            nn.Linear(pathway_dim, pathway_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(pathway_dim * 2, pathway_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(pathway_dim * 3, pathway_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pathway_dim, pathway_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _zero_loss(ref):
        return ref.sum() * 0.0

    def forward(self, wsi_bag, omic_tokens=None, omics_available=True, return_attn=False):
        batch_size, num_instances, _ = wsi_bag.shape
        wsi_tokens = self.wsi_proj(wsi_bag.float())

        if omics_available and omic_tokens is not None:
            query_tokens = omic_tokens.float()
            target_tokens = query_tokens
        else:
            query_tokens = self.pathway_queries.expand(batch_size, -1, -1)
            target_tokens = None

        v = self.v_proj(wsi_tokens)

        if self.use_pel:
            q = self.q_proj(query_tokens)
            k = self.k_proj(wsi_tokens)
            attn_logits = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.pathway_dim)
            topk = max(1, min(num_instances, int(num_instances * self.topk_ratio)))
            topk_idx = torch.topk(attn_logits, topk, dim=-1).indices
            topk_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            attn_logits = attn_logits.masked_fill(~topk_mask, torch.finfo(attn_logits.dtype).min)
            attn = F.softmax(attn_logits, dim=-1)
            pathology_evidence = torch.matmul(attn, v)
        else:
            pathology_evidence = v.mean(dim=1, keepdim=True).expand(-1, self.num_pathways, -1)
            attn = None

        recovered_evidence = self.recover(pathology_evidence)

        if target_tokens is None:
            fused_tokens = recovered_evidence
            rec_loss = self._zero_loss(recovered_evidence)
            comp_loss = self._zero_loss(recovered_evidence)
        else:
            if self.use_cgf:
                gate_input = torch.cat(
                    [target_tokens, recovered_evidence, target_tokens * recovered_evidence],
                    dim=-1
                )
                gate = self.gate(gate_input)
                fused_tokens = gate * target_tokens + (1.0 - gate) * recovered_evidence
            else:
                fused_tokens = 0.5 * (target_tokens + recovered_evidence)

            if self.use_mel:
                rec_loss = F.mse_loss(recovered_evidence, target_tokens.detach())
            else:
                rec_loss = self._zero_loss(recovered_evidence)

            if self.use_cgf:
                cosine = F.cosine_similarity(recovered_evidence, target_tokens, dim=-1)
                comp_loss = F.relu(cosine - self.comp_margin).mean()
            else:
                comp_loss = self._zero_loss(recovered_evidence)

        if return_attn:
            return fused_tokens, recovered_evidence, rec_loss, comp_loss, attn
        return fused_tokens, recovered_evidence, rec_loss, comp_loss


class PPEL(nn.Module):
    def __init__(
            self,
            args,
            omic_names=None,
            omics_input_dim=0,
    ):
        super(PPEL, self).__init__()
        omic_names = [] if omic_names is None else omic_names

        self.num_pathways = len(args.omic_sizes)
        self.num_classes = args.n_classes
        self.omic_sizes = args.omic_sizes
        self.args = args

        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim
        self.omics_input_dim = omics_input_dim
        self.bag_size = BAG_SIZE

        if omic_names:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            self.all_gene_names = list(all_gene_names)

        self.init_per_path_model(self.omic_sizes, args.omics_format)
        self.ppel = PathwayPrognosticEvidenceLearning(
            wsi_dim=self.wsi_embedding_dim,
            pathway_dim=self.wsi_projection_dim,
            num_pathways=self.num_pathways,
            topk_ratio=args.ratio_wsi,
            comp_margin=args.comp_margin,
            use_pel=args.use_pel,
            use_mel=args.use_mel,
            use_cgf=args.use_cgf,
        )

        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.wsi_projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.wsi_projection_dim, self.num_classes),
        )
        self.recovered_to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, self.wsi_projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.wsi_projection_dim, self.num_classes),
        )

        self.loss_surv = NLLSurvLoss(alpha=0.5)

    def init_per_path_model(self, omic_sizes, omics_format):
        if omics_format in ['pathways', 'groups']:
            hidden = [self.wsi_projection_dim, self.wsi_projection_dim]
            sig_networks = []
            for input_dim in omic_sizes:
                fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
                for i, _ in enumerate(hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.sig_networks = nn.ModuleList(sig_networks)
        elif omics_format == 'gene':
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=self.wsi_projection_dim)
        else:
            raise ValueError('omics_format should be pathways, gene or groups')

    def _omics_are_missing(self, x_omic):
        first = x_omic[0]
        return first is None or (isinstance(first, torch.Tensor) and torch.all(first == 0))

    def _encode_omics(self, x_omic):
        h_omic = [
            self.sig_networks[idx].forward(sig_feat)
            for idx, sig_feat in enumerate(x_omic)
        ]
        h_omic_bag = torch.stack(h_omic)
        return h_omic_bag.permute(1, 0, 2)

    def forward(self, **kwargs):
        wsi = kwargs['x_wsi']
        return_attn = kwargs["return_attn"]
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]

        is_missing_omics = self._omics_are_missing(x_omic)
        h_omic_bag = None if is_missing_omics else self._encode_omics(x_omic)

        iter_num = max(1, self.args.num_patches // self.bag_size)
        device = wsi.device
        logits = torch.empty((iter_num, wsi.size(0), self.num_classes), device=device)
        comp_loss_total = wsi.sum() * 0.0
        evidence_loss_total = wsi.sum() * 0.0
        attns = None

        for i in range(iter_num):
            idx = torch.randperm(wsi.size(1), device=device)[:self.bag_size]
            h_wsi_bag = wsi[:, idx, :]

            if return_attn:
                fused_tokens, recovered_tokens, rec_loss, comp_loss, attns = self.ppel(
                    h_wsi_bag, h_omic_bag, not is_missing_omics, return_attn=True
                )
            else:
                fused_tokens, recovered_tokens, rec_loss, comp_loss = self.ppel(
                    h_wsi_bag, h_omic_bag, not is_missing_omics, return_attn=False
                )

            fused_embed = torch.mean(fused_tokens, dim=1)
            recovered_embed = torch.mean(recovered_tokens, dim=1)
            logits[i] = self.to_logits(fused_embed)

            if self.training:
                evidence_loss = rec_loss
                if self.args.use_mel and not is_missing_omics:
                    recovered_logits = self.recovered_to_logits(recovered_embed)
                    evidence_loss = evidence_loss + F.mse_loss(recovered_logits, logits[i].detach())
                evidence_loss_total = evidence_loss_total + evidence_loss
                comp_loss_total = comp_loss_total + comp_loss

        logits = torch.mean(logits, dim=0)
        comp_loss_total = comp_loss_total / iter_num
        evidence_loss_total = evidence_loss_total / iter_num

        if return_attn:
            return logits, attns
        return logits, comp_loss_total, evidence_loss_total
