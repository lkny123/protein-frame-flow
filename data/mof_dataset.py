import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from mofdiff.common.data_utils import frac_to_cart_coords


class MOFDataset(Dataset):
    def __init__(
            self,
            *,
            cache_path,
            dataset_cfg,
            is_training,
        ):

        self.cached_data = torch.load(cache_path)

        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    @property
    def data_conf(self):
        return self._data_conf
        
    @staticmethod
    def _recenter_mof(data):
        """
        Recenters all coordinates s.t the center of mass is at the origin

        Returns:
            gt_coords: numpy array of shape (n_atoms, 3)
            gt_trans: numpy array of shape (n_bbs, 3)
            local_coords: list of numpy array local bb coordinates

        """
        def _get_cart_coords_from_bb(bb):
            cart_coords = frac_to_cart_coords(
                bb.frac_coords, 
                bb.lengths,
                bb.angles,
                bb.num_atoms
            ).numpy()
            return cart_coords

        bb_pos = []
        gt_coords = []

        for bb in data.pyg_mols:
            bb_coords = _get_cart_coords_from_bb(bb)

            bb_pos.append(np.mean(bb_coords, axis=0))
            gt_coords.append(bb_coords)
        
        bb_center = np.mean(bb_pos, axis=0)

        gt_coords = [coord - bb_center for coord in gt_coords]
        gt_trans = [pos - bb_center for pos in bb_pos]

        # get local bb coordinates
        local_coords = [coord - pos for coord, pos in zip(gt_coords, gt_trans)]

        gt_coords = np.concatenate(gt_coords, axis=0)
        gt_trans = np.stack(gt_trans, axis=0)

        return gt_coords, gt_trans, local_coords

    @staticmethod
    def _rotate_bb(bb_coord, bb_atom_type):
        """
        Returns:
            rotmats: numpy array of shape (3, 3)
            local_coord: numpy array of shape (n_bb_atoms, 3) 
        """
        rotmats = np.eye(3)
        local_coord = bb_coord

        return rotmats, local_coord 

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        """
        Returns: dictionary with following keys:
            - rotmats_1: [M, 3, 3]
            - trans_1: [M, 3]
            - res_mask: [M,]
            - diffuse_mask: [M,]
            - local_coords: [N, 3]
            - gt_coords: [N, 3]
            - bb_num_vec: [M,]
            - bb_emb: [M, 3]
            - atom_types: [N,]
            - lattice: [6,]
        """

        datapoint = self.cached_data[idx]

        # Recenter MOF
        gt_coords, gt_trans, centered_bb_coords = self._recenter_mof(datapoint)

        # Rotate MOF bbs
        gt_rotmats = []
        local_coords = []
        
        for i, bb_coord in enumerate(centered_bb_coords):
            bb_atom_type = datapoint.pyg_mols[i].atom_types
            rotmats, local_coord = self._rotate_bb(bb_coord, bb_atom_type)
            gt_rotmats.append(rotmats)
            local_coords.append(local_coord)

        gt_rotmats = np.stack(gt_rotmats, axis=0)
        local_coords = np.concatenate(local_coords, axis=0)

        # Create masks
        bb_num_vec = np.array([bb.num_atoms for bb in datapoint.pyg_mols])
        res_mask = np.ones_like(bb_num_vec)
        diffuse_mask = np.ones_like(bb_num_vec)

        # Convert numpy arrays to torch tensors
        feats = {
            'rotmats_1': torch.tensor(gt_rotmats).float(),
            'trans_1': torch.tensor(gt_trans).float(),
            'res_mask': torch.tensor(res_mask).int(),
            'diffuse_mask': torch.tensor(diffuse_mask).int(),
            'local_coords': torch.tensor(local_coords).float(),
            'gt_coords': torch.tensor(gt_coords).float(),
            'bb_num_vec': torch.tensor(bb_num_vec).int(),
            'bb_emb': torch.stack([bb.emb for bb in datapoint.bbs], dim=0).float(),
            'atom_types': torch.cat([bb.atom_types for bb in datapoint.pyg_mols], dim=0).int(),
            'lattice': torch.cat([datapoint.lengths[0], datapoint.angles[0]], dim=0).float()
        }    

        return feats