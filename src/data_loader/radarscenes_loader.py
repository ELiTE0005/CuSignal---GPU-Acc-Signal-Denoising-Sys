import h5py
import cupy as cp
import numpy as np

class RadarScenesLoader:
    def __init__(self, radar_scenes_path: str):
        self.base_path = radar_scenes_path

    def load_sequence_to_gpu(self, sequence_id: int):
        """
        Loads the radar point cloud data from a sequence directly into CuPy arrays.
        """
        h5_file = f"{self.base_path}/sequence_{sequence_id}/radar_data.h5"
        
        # We use NumPy to load from h5, then immediately transfer to GPU 
        # (h5py directly reading to cupy is currently not fully supported without intermediate numpy arrays)
        with h5py.File(h5_file, 'r') as f:
            radar_data = f['radar_data'][:]
            
            # Transfer features to GPU
            ranges = cp.asarray(radar_data['range_sc'])
            azimuths = cp.asarray(radar_data['azimuth_sc'])
            velocities = cp.asarray(radar_data['vr'])
            rcs = cp.asarray(radar_data['rcs'])
            labels = cp.asarray(radar_data['label_id'])
            
            # Optional: you can extract x_cc, y_cc for Cartesian filtering
            x_cc = cp.asarray(radar_data['x_cc'])
            y_cc = cp.asarray(radar_data['y_cc'])

            point_cloud_dict = {
                'range': ranges,
                'azimuth': azimuths,
                'velocity': velocities,
                'rcs': rcs,
                'labels': labels,
                'x': x_cc,
                'y': y_cc
            }
            
        return point_cloud_dict
