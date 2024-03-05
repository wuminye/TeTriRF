_base_ = '../default.py'
expname = 'sport1'
basedir = './logs2/'

data = dict(
	datadir='./sport_1_mask',   # scene data folder
	dataset_type='NHR',
	white_bkgd=True,
	xyz_min = [-0.1531104 , -0.99574334, -0.40296442],
	xyz_max = [0.20760746, 0.01441086, 0.49955044],
	test_frames =[5,41],
	inverse_y=True,
	load2gpu_on_the_fly=True,
)

fine_model_and_render = dict(
	num_voxels=120**3,
	num_voxels_base=120**3,
	k0_type='PlaneGrid',
	rgbnet_dim=30,
	RGB_model = 'MLP',
	rgbnet_depth = 3,
	dynamic_rgbnet = True,
	viewbase_pe = 4,
	plane_scale = 3,
)


fine_train = dict(
	N_iters=40000,
	N_rand = 17800,
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=2000,                   # count total variation loss from tv_from step
    tv_before=40000,                  # count total variation before the given number of iterations
    tv_dense_before=40000,            # count total variation densely before the given number of iterations
    weight_tv_density=4e-6,        # weight of total variation loss of density voxel grid
	weight_tv_k0=4e-6,
	weight_l1_loss=0.001,
	lrate_density=1.5e-1,           # lr of density voxel grid
    lrate_k0=1.5e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1.3e-3,            # lr of the mlp to preduct view-dependent color
	pg_scale=[1000, 2000, 3000, 4000],
    pg_scale2=[7000, 9000, 11000, 13000],
	maskout_iter = 14500,
)

coarse_train = dict(
	N_iters = 0
)