from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List, Dict, Optional, Tuple

import random
import math
import numpy as np
from tqdm.auto import tqdm

from ase import Atoms
from ase.io import write

from scipy.spatial import cKDTree 

from .core import Core, BindingSite
from .ligand import Ligand, LigandSpec
from .utils.rotation import rotation_about_axis, rotation_from_u_to_v
from .utils.geometry import farthest_point_sampling, compute_bounding_spheres, build_neighbor_map

@dataclass
class NanoCrystal:
    core: Core
    ligand_specs: List[LigandSpec]
    random_seed: int
    ligands: List[Ligand] = field(default_factory=list)

    # Rotation optimization parameters
    overlap_cutoff: float = 2.0   # Å
    coarse_step_deg: int = 18
    fine_step_deg: int = 2
    window_deg: int = 12
    active_radius_factor: float = math.sqrt(2)/2

    def __post_init__(self):
        self.core = deepcopy(self.core)
        self.binding_sites = deepcopy(self.core.binding_sites)
        self._rng = random.Random(self.random_seed)
        self.ligand_specs.sort(
            key=lambda spec: spec.ligand.volume,
            reverse=True,
        )

    def place_ligands(
        self,
        max_iters: int = 10
    ) -> None:
        
        assert not self.ligands, "Ligands have already been placed."
        
        self.ligands = []
        displaced_indices = []
        for s in self.binding_sites:
            s.passivated = False

        core_positions = self.core.atoms.get_positions()

        A_sites = [s for s in self.binding_sites if s.symbol == self.core.A]
        X_sites = [s for s in self.binding_sites if s.symbol == self.core.X]

        ligands: List[Ligand] = []
        sites: List[BindingSite] = []

        for spec in self.ligand_specs:
            lig = spec.ligand
            charge = lig.charge

            if charge > 0:
                pool = A_sites
                ligand_type = "A-site"
            else: 
                pool = X_sites
                ligand_type = "X-site"

            available = [s for s in pool if not s.passivated]

            # Coverage
            if 0.0 < spec.coverage <= 1.0:
                n_target = int(math.ceil(spec.coverage * len(available)))
            else:
                n_target = int(spec.coverage)
                if n_target > len(available):
                    print(f"[Warning] Requested {n_target} sites for ligand, "
                          f"but only {len(available)} available.")
                    n_target = len(available)

            coords = np.array([core_positions[s.index] for s in available])
            chosen_indices = farthest_point_sampling(coords, n_target, self._rng)
            chosen_sites = [available[i] for i in chosen_indices]

            for site in chosen_sites:
                lig_cloned = lig.clone()
                site.passivated = True

                ligands.append(lig_cloned)
                sites.append(site)
                displaced_indices.append(site.index)

            print(f"[Log] Placed total {len(chosen_sites)} {ligand_type} ligands.")

        n_lig = len(ligands)
        assert n_lig > 0, "No ligands to place."

        # Binding site positions and planes
        site_positions = np.array([core_positions[s.index] for s in sites])
        site_planes = np.array([s.plane for s in sites], dtype=float)

        # Core atoms coordinates except displaced ones
        n_core = core_positions.shape[0]
        core_mask = np.ones(n_core, dtype=bool)

        for idx in displaced_indices:
            if 0 <= idx < n_core:
                core_mask[idx] = False

        core_coords = core_positions[core_mask]

        if core_coords.size > 0:
            core_tree = cKDTree(core_coords)
        else:
            core_tree = None

        self._core_tree = core_tree

        # Initialize rotation angles around surface normal and current coordinates
        thetas = np.array([2.0 * math.pi * self._rng.random() for _ in range(n_lig)], dtype=float)
        ligand_coords_list: List[np.ndarray] = []
        for i in range(n_lig):
            coords_i = self._place_one_ligand(
                ligands[i],
                site_planes[i],
                site_positions[i],
                thetas[i],
            )
            ligand_coords_list.append(coords_i)

        # Rotation optimization loop
        centers, radii = compute_bounding_spheres(ligand_coords_list)
        neighbor_map: Dict[int, List[int]] = build_neighbor_map(
            centers, radii, self.overlap_cutoff
        )
        global_min, conflict_ligs, dists = self._get_conflict_ligands(
            ligand_coords_list,
            neighbor_map,
        )
        print(f"[Log] Initial global_min = {global_min:.3f} Å")

        for iter in range(1, max_iters + 1):
            if global_min >= self.overlap_cutoff:
                print(
                    f"[Log] Hard cutoff {self.overlap_cutoff:.3f} Å satisfied. Stopping optimization."
                )
                break

            active_cluster = self._build_active_cluster(conflict_ligs, site_positions)

            improved = False
            self._rng.shuffle(active_cluster)

            for i in tqdm(
                active_cluster,
                desc=f"Optimizing ligand (iter {iter})",
            ):
                neighbor_idx = neighbor_map.get(i, [])
                neighbors_coords = [ligand_coords_list[j] for j in neighbor_idx]

                d_old = dists[i]

                best_theta, best_coords = self._optimize_rotation(
                    i,
                    ligands,
                    neighbors_coords,
                    site_planes,
                    site_positions,
                )
                d_new = self._min_distance(best_coords, neighbors_coords)

                if d_new > d_old + 1e-3:
                    ligand_coords_list[i] = best_coords
                    thetas[i] = best_theta
                    improved = True

            if not improved:
                print(f"[Log] No improvement in active cluster. Stopping.")
                break

            centers, radii = compute_bounding_spheres(ligand_coords_list)
            neighbor_map = build_neighbor_map(centers, radii, self.overlap_cutoff)
            global_min, conflict_ligs, dists = self._get_conflict_ligands(
                ligand_coords_list,
                neighbor_map,
            )
            print(f"[Log] Iter {iter}  global_min = {global_min:.3f} Å")

        # Apply final coordinates to Ligand
        for lig, coords in zip(ligands, ligand_coords_list):
            lig.atoms.set_positions(coords)
            self.ligands.append(lig)

        # Update core by removing displaced atoms           
        core_atoms = self.core.atoms
        core_symbols = core_atoms.get_chemical_symbols()
        core_positions = core_atoms.get_positions()

        core_mask = np.ones(len(core_symbols), dtype=bool)
        for idx in displaced_indices:
            if 0 <= idx < len(core_mask):
                core_mask[idx] = False

        stripped_symbols = [s for s, keep in zip(core_symbols, core_mask) if keep]
        stripped_positions = core_positions[core_mask]

        self.core.atoms = Atoms(
            symbols=stripped_symbols,
            positions=stripped_positions,
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )

    def _min_distance(
        self,
        coords_i: np.ndarray,
        neighbors_coords: Optional[List[np.ndarray]],
    ) -> float:
        
        # Distance to core
        core_tree = self._core_tree
        if core_tree is not None:
            dists, _ = core_tree.query(coords_i)
            d_core = float(np.min(dists))
        else:
            d_core = float("inf")

        if d_core < self.overlap_cutoff:
            return d_core
        
        if not neighbors_coords:
            return d_core
        
        # Distance to neighboring ligands
        neighbors = np.vstack(neighbors_coords)
        diff = coords_i[None, :, :] - neighbors[:, None, :]
        d2 = np.sum(diff * diff, axis=-1)
        d_lig = float(np.sqrt(d2.min()))

        return min(d_core, d_lig)

    def _get_conflict_ligands(
        self,
        ligand_coords_list: List[np.ndarray],
        neighbor_map: Dict[int, List[int]],
    ) -> Tuple[float, List[int], np.ndarray]:

        n_lig = len(ligand_coords_list)
        conflict_ligs = []
        dists = np.empty(n_lig, dtype=float)
        global_min = float("inf")

        for i in range(n_lig):
            coords_i = ligand_coords_list[i]
            idxs = neighbor_map.get(i, [])

            neighbors_coords = [ligand_coords_list[j] for j in idxs] if idxs else None

            d_i = self._min_distance(
                coords_i,
                neighbors_coords=neighbors_coords,
            )
            dists[i] = d_i
            global_min = min(global_min, d_i)

            if d_i < self.overlap_cutoff:
                conflict_ligs.append(i)

        return global_min, conflict_ligs, dists

    def _build_active_cluster(
        self,
        conflict_ligs: List[int],
        site_positions: np.ndarray,
    ) -> List[int]:

        radius = self.active_radius_factor * self.core.a + 1e-2

        if not conflict_ligs:
            return []

        site_tree = cKDTree(site_positions)

        active = set(conflict_ligs)  # start from conflict ligands

        # BFS-style cluster expansion
        queue = list(conflict_ligs)
        visited = set(conflict_ligs)

        while queue:
            i = queue.pop()
            pos_i = site_positions[i]

            neighbor_js = site_tree.query_ball_point(pos_i, r=radius)
            for j in neighbor_js:
                if j in visited:
                    continue
                visited.add(j)
                active.add(j)
                queue.append(j)

        return sorted(list(active))

    def _place_one_ligand(
        self,
        ligand: Ligand,
        plane: np.ndarray,
        site_pos: np.ndarray,
        theta: float,
    ) -> np.ndarray:

        coords_loc = ligand.atoms.get_positions()

        # Rotate to surface normal
        R_align = rotation_from_u_to_v(
            np.array([0.0, 0.0, 1.0], dtype=float),
            plane,
        )
        coords_aligned = coords_loc @ R_align.T

        # Rotate around surface normal 
        R_rot = rotation_about_axis(plane, theta)
        coords_rot = coords_aligned @ R_rot.T

        # Translate to binding site position
        coords_final = coords_rot + site_pos

        return coords_final

    def _optimize_rotation(
        self,
        i: int,
        ligands: List[Ligand],
        neighbors_coords: List[np.ndarray],
        site_planes: np.ndarray, 
        site_positions: np.ndarray,
    ) -> Tuple[float, np.ndarray]:

        cutoff = self.overlap_cutoff
        ligand_i = ligands[i]

        # Search parameters
        coarse_step_deg = self.coarse_step_deg
        fine_step_deg = self.fine_step_deg
        window_deg = self.window_deg

        # Initialization
        best_theta = None
        best_min_d = -float("inf")
        best_coords_i = None
        theta0 = 2.0 * math.pi * self._rng.random()
        
        # Coarse search
        n_coarse = max(int(round(360.0 / coarse_step_deg)), 1)

        for k in range(n_coarse):
            theta = theta0 + 2.0 * math.pi * (k / n_coarse)
            theta = theta % (2.0 * math.pi)

            coords_i = self._place_one_ligand(
                ligands[i],
                site_planes[i],
                site_positions[i],
                theta,
            )
            min_d = self._min_distance(coords_i, neighbors_coords)

            better = False
            if best_theta is None:
                better = True
            else:
                # Case 1: both new and best are below cutoff
                if best_min_d < cutoff and min_d > best_min_d:
                    better = True
                # Case 2: best is below cutoff, new candidate is above cutoff
                elif best_min_d < cutoff and min_d >= cutoff:
                    better = True
                # Case 3: both new and best are above cutoff, prefer larger min_d
                elif best_min_d >= cutoff and min_d >= cutoff and min_d > best_min_d:
                    better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        # Fine search 
        window_rad = math.radians(window_deg)
        fine_step_rad = math.radians(fine_step_deg)

        n_fine = int(round(2.0 * window_rad / fine_step_rad)) + 1
        for k in range(n_fine):
            delta = -window_rad + k * (2.0 * window_rad / (n_fine - 1))
            theta = (best_theta + delta) % (2.0 * math.pi)

            coords_i = self._place_one_ligand(ligand_i, site_planes[i], site_positions[i], theta)
            min_d = self._min_distance(coords_i, neighbors_coords)

            better = False
            if best_min_d < cutoff and min_d > best_min_d:
                better = True
            elif best_min_d < cutoff and min_d >= cutoff:
                better = True
            elif best_min_d >= cutoff and min_d >= cutoff and min_d > best_min_d:
                better = True

            if better:
                best_theta = theta
                best_min_d = min_d
                best_coords_i = coords_i

        return best_theta, best_coords_i

    def to(self, fmt: str = "xyz", filename: str = None):
        at = self.atoms
        formula = at.get_chemical_formula()
        write(filename, at, format=fmt, comment=formula)

    @property
    def atoms(self) -> Atoms:
        core_atoms = self.core.atoms
        core_symbols = list(core_atoms.get_chemical_symbols())
        core_positions = core_atoms.get_positions()

        all_symbols = list(core_symbols)
        all_positions = [core_positions]

        for lig in self.ligands:
            lig_atoms = lig.atoms
            all_symbols.extend(lig_atoms.get_chemical_symbols())
            all_positions.append(lig_atoms.get_positions())

        all_positions = np.vstack(all_positions)

        return Atoms(
            symbols=all_symbols,
            positions=all_positions,
            pbc=core_atoms.pbc,
            cell=core_atoms.get_cell(),
        )