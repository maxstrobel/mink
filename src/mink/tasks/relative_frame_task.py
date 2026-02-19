"""Relative frame task implementation."""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..lie import SE3
from .task import Objective, Task

try:
    if os.environ.get("MINK_DISABLE_NATIVE", ""):
        raise ImportError
    from ..lie import _lie_ops_c as _native  # noqa: PLC0415
except ImportError:
    _native = None  # type: ignore[assignment]


class RelativeFrameTask(Task):
    """Regulate the pose of a frame relative to another frame.

    Attributes:
        frame_name: Name of the frame to regulate, typically the name of body, geom
            or site in the robot model.
        frame_type: The frame type: `body`, `geom` or `site`.
        root_name: Name of the frame the task is relative to.
        root_type: The root frame type: `body`, `geom` or `site`.
        transform_target_to_root: Target pose in the root frame.
    """

    k: int = 6
    transform_target_to_root: SE3 | None

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        root_name: str,
        root_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.root_name = root_name
        self.root_type = root_type
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.transform_target_to_root = None

        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        position_cost = np.atleast_1d(position_cost)
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                "1 (aka identical cost for all coordinates) or (3,) but got "
                f"{position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        orientation_cost = np.atleast_1d(orientation_cost)
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                "shape 1 (aka identical cost for all coordinates) or (3,) but got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[3:] = orientation_cost

    def set_target(self, transform_target_to_root: SE3) -> None:
        """Set the target pose in the root frame.

        Args:
            transform_target_to_root: Transform from the task target frame to the
                root frame.
        """
        self.transform_target_to_root = transform_target_to_root.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(
            configuration.get_transform(
                self.frame_name,
                self.frame_type,
                self.root_name,
                self.root_type,
            )
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        transform_frame_to_root = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )
        if _native is not None:
            return _native.se3_rminus(
                transform_frame_to_root.wxyz_xyz,
                self.transform_target_to_root.wxyz_xyz,
            )
        return transform_frame_to_root.rminus(self.transform_target_to_root)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )

        transform_frame_to_root = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )

        if _native is not None:
            target = self.transform_target_to_root.wxyz_xyz
            frame = transform_frame_to_root.wxyz_xyz
            T_ft = _native.se3_inverse_multiply(target, frame)
            # Adjoint of frame_to_root inverse for the jacobian difference.
            inv_frame = transform_frame_to_root.inverse()
            A_rf = _native.se3_adjoint(inv_frame.wxyz_xyz)
            return _native.se3_jlog(T_ft) @ (
                jacobian_frame_in_frame - A_rf @ jacobian_root_in_root
            )
        else:
            transform_frame_to_target = (
                self.transform_target_to_root.inverse() @ transform_frame_to_root
            )
            return transform_frame_to_target.jlog() @ (
                jacobian_frame_in_frame
                - transform_frame_to_root.inverse().adjoint() @ jacobian_root_in_root
            )

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute shared quantities once.
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        jacobian_root_in_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )
        transform_frame_to_root = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )

        if _native is not None:
            target = self.transform_target_to_root.wxyz_xyz
            frame = transform_frame_to_root.wxyz_xyz
            error = _native.se3_rminus(frame, target)
            T_ft = _native.se3_inverse_multiply(target, frame)
            inv_frame = transform_frame_to_root.inverse()
            A_rf = _native.se3_adjoint(inv_frame.wxyz_xyz)
            jacobian = _native.se3_jlog(T_ft) @ (
                jacobian_frame_in_frame - A_rf @ jacobian_root_in_root
            )
        else:
            error = transform_frame_to_root.rminus(self.transform_target_to_root)
            transform_frame_to_target = (
                self.transform_target_to_root.inverse() @ transform_frame_to_root
            )
            jacobian = transform_frame_to_target.jlog() @ (
                jacobian_frame_in_frame
                - transform_frame_to_root.inverse().adjoint() @ jacobian_root_in_root
            )

        minus_gain_error = -self.gain * error
        weight = np.diag(self.cost)
        weighted_jacobian = weight @ jacobian
        weighted_error = weight @ minus_gain_error
        mu = self.lm_damping * weighted_error @ weighted_error
        eye_tg = np.eye(configuration.model.nv)
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return Objective(H, c)
