_base_ = '../default.py'
expname = 'kpop'
basedir = './logs2/'

data = dict(
	datadir='./kpop',
	dataset_type='NHR',
	white_bkgd=True,
	xyz_min = [-0.4,  -0.3, -1.15 ],
    xyz_max = [ 0.4,  0.6,  0.05],
	test_frames =[6,39],
	inverse_y=True,
	load2gpu_on_the_fly=True,
	width=1280,                   # enforce image width
    height=720,                  # enforce image height

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
)


fine_train = dict(
	N_iters=40000,
	N_rand = 16800,
	tv_every=1,                   # count total variation loss every tv_every step
    tv_after=2000,                   # count total variation loss from tv_from step
    tv_before=40000,                  # count total variation before the given number of iterations
    tv_dense_before=40000,            # count total variation densely before the given number of iterations
    weight_tv_density=3e-6,        # weight of total variation loss of density voxel grid
	weight_tv_k0=2e-6,
	weight_l1_loss=0.001,
	lrate_density=1.5e-1,           # lr of density voxel grid
    lrate_k0=1.4e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1.1e-3,            # lr of the mlp to preduct view-dependent color
	pg_scale=[1000, 2000, 3000, 4000],
    pg_scale2=[7000, 9000, 11000, 13000],
	maskout_iter = 14500,
)

coarse_train = dict(
	N_iters = 0
)
