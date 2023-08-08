from typing import List, Union
from lib.ModelType import ModelMetaData, TrainingData
import schnetpack as spk
import torch
import torch.nn as nn
import torch.nn.functional as F
import ase

def transform(batch_data: Union[TrainingData, List[TrainingData]], embedding_fn: spk.representation.SchNet):
        # coords: unit Angstron
        if type(batch_data) != list:
            batch_data = [batch_data]
        
        converter = spk.AtomsConverter()
        new_batch_data = []
        coords_mask_list = []
        for data in batch_data:
            atoms = data.atoms
            coords = data.coords
            ase_atoms = ase.Atoms()
            for atom, coord in zip(atoms, coords.reshape(-1, 3)):
                ase_atoms.append(ase.Atom(atom, coord))
            atoms_features =  embedding_fn(converter(ase_atoms)).squeeze()

            _pre_features = torch.tensor(data.pre_features, dtype=torch.float32)

            coords_mask = _pre_features[:, -1]
            encode = _pre_features[:, :3]
            indices = _pre_features[:, 3:3+4].unsqueeze(0).long()
            features = _pre_features[:, 3+4:-1]
            mask = (indices > -1).long()
            _indices = indices.squeeze().unsqueeze(-1)
            _features = atoms_features[_indices].squeeze(-2) * mask.squeeze().unsqueeze(-1)
            primitives_features = _features.sum(-2)
            new_data = torch.hstack([
                    encode,
                    primitives_features,
                    features,
                ]).unsqueeze(0)
            new_batch_data.append(
                new_data
            )
            coords_mask_list.append(coords_mask.unsqueeze(0))

        return torch.cat(new_batch_data), torch.cat(coords_mask_list)

class ModelActor(nn.Module):
    def __init__(self, CustomModel: nn.Module, meta_data: ModelMetaData):
        super(ModelActor, self).__init__()
        self.device = meta_data.device
        self.embedding_fn = spk.representation.SchNet(**vars(meta_data.schnet_metadata))
        self.custom_model = CustomModel(
            meta_data.input_size + (meta_data.schnet_metadata.n_atom_basis - 4),
            meta_data.act_size,
            meta_data.hidden_size,
            meta_data.n_head,
        )
        
        self.logstd = nn.Parameter(torch.zeros(meta_data.act_size))

    def forward(self, batch_data: List[TrainingData]):
        obs, mask = transform(batch_data, self.embedding_fn)
        obs.to(self.device)
        mask.to(self.device)

        out = self.custom_model(obs, mask)
        out = torch.tanh(out)
        return out

class ModelCritic(nn.Module):
    def __init__(self, CustomModel: nn.Module, meta_data: ModelMetaData):
        super(ModelCritic, self).__init__()
        self.embedding_fn = spk.representation.SchNet(**vars(meta_data.schnet_metadata))
        self.device = meta_data.device
        self.custom_model = CustomModel(
            meta_data.input_size + (meta_data.schnet_metadata.n_atom_basis - 4),
            meta_data.crt_size,
            meta_data.hidden_size,
            meta_data.n_head,
        )

    def forward(self, batch_data: List[TrainingData]):
        obs, mask = transform(batch_data, self.embedding_fn)
        obs.to(self.device)
        mask.to(self.device)

        out = self.custom_model(obs, mask)
        out = torch.sum(out*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)
        
        return out

class ModelSACTwinQ(nn.Module):
    def __init__(self, CustomModel: nn.Module, meta_data: ModelMetaData):
        super(ModelSACTwinQ, self).__init__()
        self.device = meta_data.device
        self.embedding_fn = spk.representation.SchNet(**vars(meta_data.schnet_metadata))
        sac_input_size = meta_data.input_size + meta_data.act_size + (meta_data.schnet_metadata.n_atom_basis - 4)
        self.custom_model_1 = CustomModel(
            sac_input_size,
            meta_data.crt_size,
            meta_data.hidden_size,
            meta_data.n_head,
        )

        self.custom_model_2 = CustomModel(
            sac_input_size,
            meta_data.crt_size,
            meta_data.hidden_size,
            meta_data.n_head,
        )

    def forward(self, batch_data: List[TrainingData], act):
        obs, mask = transform(batch_data, self.embedding_fn)
        obs.to(self.device)
        mask.to(self.device)
       
        x1 = torch.cat([obs, act], dim=-1)
        x2 = torch.cat([obs, act], dim=-1)

        out1 = self.custom_model_1(x1, mask)
        out1 = torch.sum(out1*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)

        
        out2 = self.custom_model_2(x2, mask)        
        out2 = torch.sum(out2*mask.unsqueeze(-1), dim = -2) / torch.sum(mask, dim = -1).unsqueeze(-1)
        
        return out1, out2