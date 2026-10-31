"""Explicit CA-only interface for downstream dynamics experiments.

Future encoding is an oracle operation. Forecast callers must supply generated H
and the observed anchor H separately. No atom37 output is implied.
"""
from types import SimpleNamespace
import torch
from torch import nn
from protein_jepa_frame.models.ca_geometry import project_ca_bonds
from protein_jepa_frame.training.train_lasp_calibration import Calibration


class CAComponentBundle(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        source=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if source.get('checkpoint_kind')!='ca_component_probe':
            raise ValueError('requires an explicitly labelled CA component checkpoint')
        args=SimpleNamespace(**source['calibration_args'])
        args.init_from=checkpoint
        model=Calibration(args,device)
        self.encoder=model.encoder
        self.decoder=model.decoder
        self.projection_steps=args.bond_projection_steps
        self.requires_dynamics_recalibration=True
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.to(device).eval()

    def encode(self,backbone,mask,aatype,residue_index,chain_id):
        return self.encoder(backbone[:,None],mask[:,None],aatype,residue_index,chain_id,
                            token_mask=mask[:,None])[:,0]

    def decode_ca(self,generated_h,anchor_h,aatype,residue_index,chain_id,rotation,anchor,mask):
        ca=self.decoder(generated_h,anchor_h,aatype,residue_index,chain_id,rotation,anchor,mask)
        if self.projection_steps:
            ca=project_ca_bonds(ca,anchor,mask,residue_index,chain_id,self.projection_steps)
        return ca
