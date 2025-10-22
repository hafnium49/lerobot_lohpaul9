import sys
sys.path.insert(0, 'src')
import mujoco as mj

# Load model directly
model = mj.MjModel.from_xml_path('src/lerobot/envs/so101_assets/paper_square_realistic.xml')

# Print ALL geoms
print('ALL geoms in model:')
print('-' * 100)
print(f"{'Name':50s} | contype | conaffinity | group | type")
print('-' * 100)
for i in range(model.ngeom):
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f'geom_{i}'
    contype = model.geom_contype[i]
    conaffinity = model.geom_conaffinity[i]
    group = model.geom_group[i]
    geom_type = model.geom_type[i]

    print(f'{geom_name:50s} | {contype:7d} | {conaffinity:11d} | {group:5d} | {geom_type:4d}')
