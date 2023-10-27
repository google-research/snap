"""Useful types."""

import dataclasses
import enum
import os.path as osp
from typing import Any, Dict, NamedTuple, Optional, Tuple

from etils.array_types import BoolArray
import tensorflow_datasets as tfds

DataDict = Dict[str, Any]  # data for a given segment, scene, camera, etc.
SegmentsDict = Dict[str, DataDict]  # a collection of segments
RastersDict = dict[str, BoolArray['H, W']]


# A valid plane height is always positive w.r.t. the scene coordinate system.
INVALID_GROUND_PLANE_HEIGHT = -1.0

# Semantic classes
AERIAL_BUILDING_CLASSES = ['buildings_raw', 'buildings_contoured']
SURFEL_ROAD_CLASSES = [
    'crosswalk',
    'sidewalk',
    'pavedroad',
    'stopline',
    'line',
    'otherlanemarking',
]


class DataMode(str, enum.Enum):
  SINGLE_SCENE = 'single_scene'
  PAIR_SCENES = 'pair_scenes'
  PAIR_SCENE_VIEW = 'pair_scene_view'


@dataclasses.dataclass
class SceneConfig:
  """Configuration for scene (grid and view) selection."""

  grid_size: Tuple[int, int, int] = (24, 32, 12)
  grid_z_offset: int = 4
  center_grid_around_reference: bool = True
  num_views: int = 10
  min_distance_between_views: float = 1.5
  max_distance_between_views: float = 15
  only_views_in_grid: bool = True
  reference_cameras: Tuple[str, ...] = ('side_left', 'side_right')
  reference_vehicles: Tuple[str, ...] = ('CAR',)
  constrain_all_cameras: bool = True
  single_segment_add_front_rear_cameras: bool = True
  single_segment_add_front_rear_cameras_every: Optional[int] = 3
  streetview_hfov_deg: float = 72.0
  camera_frustum_depth: float = 16.0


@dataclasses.dataclass
class PairingConfig:
  """Configuration for pairing scenes."""

  min_overlap: float = 0.3
  max_overlap: float = 0.7
  min_distance_to_scene_views: Optional[float] = None
  max_elevation_diff: float = 2.0
  num_queries_per_scene: Optional[int] = None
  ratio_trekker: float = 0.5


@dataclasses.dataclass
class ProcessingConfig(tfds.core.BuilderConfig):
  """Configuration for the entire data processing pipeline."""

  data_path: Optional[str] = None
  scenes_sstable_path: str = ''
  frames_sstable_path: str = ''
  s2_cell_list_path: str = ''

  image_downsampling_factor: Optional[int] = None
  pose_tag: Optional[str] = None

  split_by_s2_cell: bool = True
  generate_training_split: bool = True
  max_total_area_km2: Optional[float] = None
  evaluation_num_cells: Optional[int] = 5
  evaluation_s2_level: int = 14
  evaluation_max_num_examples: Optional[int] = None

  scene_types: Tuple[str, ...] = ('OUTDOOR',)
  vehicle_types: Tuple[str, ...] = ('CAR', 'TREKKER')
  vehicle_types_for_map: Optional[Tuple[str, ...]] = ('CAR',)
  bin_level: int = 18

  single_segment_per_scene: bool = True
  min_num_runs_per_scene: int = 2
  min_num_segments_per_vehicle: int = 1
  scene_config: SceneConfig = dataclasses.field(default_factory=SceneConfig)

  mode: DataMode = DataMode.SINGLE_SCENE
  pairing_config: PairingConfig = dataclasses.field(
      default_factory=PairingConfig
  )

  @property
  def need_lidar_semantics(self) -> bool:
    return (
        self.rasters_config.add_gt_semantics
        or self.lidar_config.add_gt_semantics
    )

  @classmethod
  def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
    if config_dict.pop('pair_scenes', False):
      config_dict['mode'] = DataMode.PAIR_SCENES
    elif 'mode' in config_dict:
      config_dict['mode'] = DataMode(config_dict['mode'])
    config_dict = {
        **config_dict,
        'scene_config': SceneConfig(**config_dict.get('scene_config', {})),
        'rasters_config': RastersConfig(
            **config_dict.get('rasters_config', {})
        ),
        'pairing_config': PairingConfig(
            **config_dict.get('pairing_config', {})
        ),
    }
    return cls(**config_dict)
