"""Collision avoidance limit."""

import itertools
from typing import Sequence

import mujoco
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Type aliases.
Geom = int | str
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]


def compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1_id: int,
    geom2_id: int,
    fromto: np.ndarray,
    normal: np.ndarray,
    jac1: np.ndarray,
    jac2: np.ndarray,
) -> np.ndarray:
    """Compute the contact normal Jacobian between two geoms."""
    normal[:] = fromto[3:] - fromto[:3]
    mujoco.mju_normalize3(normal)
    geom_bodyid = model.geom_bodyid
    mujoco.mj_jac(model, data, jac2, None, fromto[3:], geom_bodyid[geom2_id])
    mujoco.mj_jac(model, data, jac1, None, fromto[:3], geom_bodyid[geom1_id])
    jac2 -= jac1
    return normal @ jac2


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Returns true if the geoms are part of the same body, or if their bodies are
    welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # body_weldid is the ID of the body's weld.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs in which to perform active collision
            avoidance. A collision pair is defined as a pair of geom groups. A geom
            group is a set of geom names. For each geom pair, the solver will
            attempt to compute joint velocities that avoid collisions between every
            geom in the first geom group with every geom in the second geom group.
            Self collision is achieved by adding a collision pair with the same
            geom group in both pair fields.
        gain: Gain factor in (0, 1] that determines how fast the geoms are
            allowed to move towards each other at each iteration. Smaller values
            are safer but may make the geoms move slower towards each other.
        minimum_distance_from_collisions: The minimum distance to leave between
            any two geoms. A negative distance allows the geoms to penetrate by
            the specified amount.
        collision_detection_distance: The distance between two geoms at which the
            active collision avoidance limit will be active. A large value will
            cause collisions to be detected early, but may incur high computational
            cost. A negative value will cause the geoms to be detected only after
            they penetrate by the specified amount.
        bound_relaxation: An offset on the upper bound of each collision avoidance
            constraint.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.

        Args:
            model: MuJoCo model.
            geom_pairs: Set of collision pairs in which to perform active collision
                avoidance. A collision pair is defined as a pair of geom groups. A geom
                group is a set of geom names. For each collision pair, the mapper will
                attempt to compute joint velocities that avoid collisions between every
                geom in the first geom group with every geom in the second geom group.
                Self collision is achieved by adding a collision pair with the same
                geom group in both pair fields.
            gain: Gain factor in (0, 1] that determines how fast the geoms are
                allowed to move towards each other at each iteration. Smaller values
                are safer but may make the geoms move slower towards each other.
            minimum_distance_from_collisions: The minimum distance to leave between
                any two geoms. A negative distance allows the geoms to penetrate by
                the specified amount.
            collision_detection_distance: The distance between two geoms at which the
                active collision avoidance limit will be active. A large value will
                cause collisions to be detected early, but may incur high computational
                cost. A negative value will cause the geoms to be detected only after
                they penetrate by the specified amount.
            bound_relaxation: An offset on the upper bound of each collision avoidance
                constraint.
        """
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)

        self._fromto = np.empty(6)
        self._normal = np.empty(3)
        self._jac1 = np.empty((3, model.nv))
        self._jac2 = np.empty((3, model.nv))

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        model = self.model
        data = configuration.data
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, model.nv))
        fromto = self._fromto
        normal = self._normal
        jac1 = self._jac1
        jac2 = self._jac2
        distmax = self.collision_detection_distance
        min_dist = self.minimum_distance_from_collisions
        gain = self.gain
        relaxation = self.bound_relaxation
        for idx, (geom1_id, geom2_id) in enumerate(self.geom_id_pairs):
            dist = mujoco.mj_geomDistance(
                model, data, geom1_id, geom2_id, distmax, fromto
            )
            if abs(dist - distmax) < 1e-12:
                continue
            row = compute_contact_normal_jacobian(
                model, data, geom1_id, geom2_id, fromto, normal, jac1, jac2
            )
            if dist > min_dist:
                upper_bound[idx] = (gain * (dist - min_dist) / dt) + relaxation
            else:
                upper_bound[idx] = relaxation
            sign = -1.0 if dist >= 0 else 1.0
            coefficient_matrix[idx] = sign * row
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> list[int]:
        """Take a heterogeneous list of geoms (specified via ID or name) and return
        a homogenous list of IDs (int)."""
        list_of_int: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                list_of_int.append(g)
            else:
                assert isinstance(g, str)
                list_of_int.append(self.model.geom(g).id)
        return list_of_int

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_A = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_B = self._homogenize_geom_id_list(collision_pair[1])
            id_pair_A = list(set(id_pair_A))
            id_pair_B = list(set(id_pair_B))
            geom_id_pairs.append((id_pair_A, id_pair_B))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs):
        """Returns a set of geom ID pairs for all possible geom-geom collisions.

        The contacts are added based on the following heuristics:
            1) Geoms that are part of the same body or weld are not included.
            2) Geoms where the body of one geom is a parent of the body of the other
                geom are not included.
            3) Geoms that fail the contype-conaffinity check are ignored.

        Note:
            1) If two bodies are kinematically welded together (no joints between them)
                they are considered to be the same body within this function.
        """
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                weld_body_cond = not _is_welded_together(self.model, geom_a, geom_b)
                parent_child_cond = not _are_geom_bodies_parent_child(
                    self.model, geom_a, geom_b
                )
                contype_conaffinity_cond = _is_pass_contype_conaffinity_check(
                    self.model, geom_a, geom_b
                )
                if weld_body_cond and parent_child_cond and contype_conaffinity_cond:
                    geom_id_pairs.append((min(geom_a, geom_b), max(geom_a, geom_b)))
        # Deduplicate pairs in case of overlapping geom groups.
        return list(set(geom_id_pairs))
