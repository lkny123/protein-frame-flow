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
            ).numpy().astype(np.float64)
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
    def _get_equiv_vec(cart_coords, atom_types):

        centroid = np.mean(cart_coords, axis=0)

        # Center of mass weighted by atomic number
        weight = atom_types / atom_types.sum()
        weighted_centroid = np.sum(cart_coords * weight[:, None], axis=0)

        # Equivariant vector
        equiv_vec = weighted_centroid - centroid

        # If v = 0 (symmetric), take the closest non-zero atom
        if np.allclose(equiv_vec, 0):
            dist = np.linalg.norm(cart_coords, axis=1)
            sorted_indices = np.argsort(dist)

            i = 0
            while i < len(sorted_indices) and np.allclose(equiv_vec, 0):
                equiv_vec = cart_coords[sorted_indices[i]]
                i += 1
        
        assert not np.allclose(equiv_vec, 0), "Equivariant vector is zero"
        return equiv_vec
    
    @staticmethod
    def _get_pca_axes(data):
        # Center the data
        data_mean = np.mean(data, axis=0)
        centered_data = data - data_mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        if covariance_matrix.ndim == 0:
            return np.zeros(3), np.eye(3)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors    

    def _get_equivariant_axes(self, cart_coords, atom_types):
        """
        Return:
            R: equivariant rotation matrix
        """

        if cart_coords.shape[0] == 1:
            return np.eye(3)

        equiv_vec = self._get_equiv_vec(cart_coords, atom_types)    # v(X)

        _, axes = self._get_pca_axes(cart_coords)                   # PCA(X)
        ve = equiv_vec @ axes
        flips = ve < 0 
        axes = np.where(flips[None], -axes, axes)

        right_hand = np.stack(
            [axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1
        )
        
        return right_hand
    
    def _rotate_bb(self, bb_coord, bb_atom_type):
        """
        Returns:
            rotmats: numpy array of shape (3, 3)
            local_coord: numpy array of shape (n_bb_atoms, 3) 
        """
        rotmats = self._get_equivariant_axes(bb_coord, bb_atom_type) # f(X)
        local_coord = bb_coord @ rotmats                             # g(X) = X f(X)

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
            bb_atom_type = datapoint.pyg_mols[i].atom_types.numpy()
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