from __future__ import annotations
from collections.abc import Generator
from dataclasses import dataclass
import pathlib

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SectionConfig:
    index: int
    length: int
    dtype: jnp.dtype


class Mesh:
    VALID_HEADERS: dict[str, SectionConfig] = {
        "*nodes": SectionConfig(0, 3, jnp.float32),
        "*edges": SectionConfig(1, 2, jnp.int32),
        "*triangles": SectionConfig(2, 3, jnp.int32),
    }

    def __init__(self, nodes: jax.Array, edges: jax.Array, triangles: jax.Array):
        self._nodes = nodes
        self._edges = edges
        self._triangles = triangles
        self._rod_edges, self._joint_edges = self.separate_joint_edges(edges, triangles)

    @classmethod
    def from_txt(cls, fname: str | pathlib.Path) -> Mesh:
        if not (fname := pathlib.Path(fname)).is_file():
            raise ValueError(f"{fname} is not a valid file path")

        sections: list[list[list[float]]] = [[] for _ in range(len(cls.VALID_HEADERS))]
        seen_headers: set[str] = set()
        config: SectionConfig | None = None

        for line_num, line in cls._iter_lines(fname):
            if line.startswith("*"):
                config = cls._parse_header(line_num, line, seen_headers)
            else:
                if config is None:
                    raise ValueError(f"Line {line_num}: Data found before any header")
                sections[config.index].append(
                    cls._parse_data_line(line_num, line, config)
                )

        arrays = []
        for cfg, data in zip(cls.VALID_HEADERS.values(), sections):
            if not data:
                arr = jnp.empty((0, cfg.length), dtype=cfg.dtype)
            else:
                arr = jnp.array(data, dtype=cfg.dtype)
                if cfg.dtype == jnp.int32:
                    arr -= 1  # Matlab -> 0-based
            arrays.append(arr)

        return cls(*arrays)

    @staticmethod
    def _iter_lines(fname: pathlib.Path) -> Generator[tuple[int, str]]:
        with open(fname, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line_num, line

    @classmethod
    def _parse_header(cls, line_num: int, line: str, seen: set[str]) -> SectionConfig:
        header = line.lower()
        if header not in cls.VALID_HEADERS:
            raise ValueError(f"Line {line_num}: Unknown header '{line}'")
        if header in seen:
            raise ValueError(f"Line {line_num}: Duplicate header '{line}'")
        seen.add(header)
        return cls.VALID_HEADERS[header]

    @staticmethod
    def _parse_data_line(
        line_num: int, line: str, config: SectionConfig
    ) -> list[float]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != config.length:
            raise ValueError(
                f"Line {line_num}: Expected {config.length} values, got {len(values)}"
            )
        try:
            return [float(v) for v in values]
        except ValueError as e:
            raise ValueError(
                f"Line {line_num}: Invalid numeric value in '{line}'"
            ) from e

    @staticmethod
    def separate_joint_edges(
        edges: jax.Array, triangles: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Separate rod-shell joint edges from rod-rod edges.

        Args:
            edges (jax.Array):
            triangles (jax.Array):

        Returns:
            rod_rod_edges (jax.Array): The rod-rod edges.
            rod_shell_edges (jax.Array): The rod-shell edges.
        """
        shell_nodes = jnp.unique(triangles)
        in_shell0 = jnp.isin(edges[:, 0], shell_nodes)
        in_shell1 = jnp.isin(edges[:, 1], shell_nodes)
        is_joint_edge = in_shell0 | in_shell1
        joint_edges = edges[is_joint_edge]
        non_joint_edges = edges[~is_joint_edge]
        mask = jnp.isin(joint_edges[:, 0], shell_nodes)
        swapped_edges = jnp.where(
            mask[:, None],
            joint_edges[:, ::-1],
            joint_edges,
        )
        return non_joint_edges, swapped_edges
