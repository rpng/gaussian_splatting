import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

#python3 ../../run_colmap.py --run_undistort --images /home/saimouli/Documents/ov_nerf_slam/gaussian_splatting/data/euroc_test/images -
# -out /home/saimouli/Documents/ov_nerf_slam/gaussian_splatting/data/euroc_test/transforms.json

def parse_args():
	parser = argparse.ArgumentParser(
	    description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--run_colmap", action="store_true",
	                    help="run colmap first on the image folder")
	parser.add_argument("--run_undistort", action="store_true",
	                    help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive", "sequential", "spatial", "transitive",
	                    "vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
	parser.add_argument("--colmap_db", default="colmap.db",
	                    help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL",
	                    "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
	parser.add_argument("--colmap_camera_params", default="",
	                    help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images",
	                    help="Input path to the images.")
	parser.add_argument("--text", default="colmap_text",
	                    help="Input path to the colmap text files (set automatically if --run_colmap is used).")
	parser.add_argument("--out", default="transforms.json", help="Output path.")
	parser.add_argument("--overwrite", action="store_true",
	                    help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
	args = parser.parse_args()
	return args


def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


def run_colmap(args):
	colmap_binary = "colmap"

	db = args.colmap_db
	images = "\"" + args.images + "\""
	db_noext = str(Path(db).with_suffix(""))

	if args.text == "text":
		args.text = db_noext+"_text"
	text = args.text
	sparse = db_noext+"_sparse"
	print(
	    f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	if not args.overwrite and (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	match_cmd = f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(
	    f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(
	    f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def undistort(args):
	## Image undistortion
	# We need to undistort our images into ideal pinhole intrinsics.
	colmap_binary = "colmap"
	db = args.colmap_db
	db_noext = str(Path(db).with_suffix(""))
	sparse_undist = db_noext+"_undistort"
	sparse = db_noext+"_sparse"

	text_undist = sparse_undist+"_text"

	try:
		shutil.rmtree(sparse_undist)
	except:
		pass

	do_system(f"mkdir {sparse_undist}")
	do_system(
		f"{colmap_binary} image_undistorter --image_path {args.images} --input_path {sparse}/0 --output_path {sparse_undist}/0 --output_type COLMAP"
	)

	try:
		shutil.rmtree(text_undist)
	except:
		pass

	do_system(f"mkdir {text_undist}")
	do_system(
	    f"{colmap_binary} model_converter --input_path {sparse_undist}/0/sparse --output_path {text_undist} --output_type TXT")
	
	return text_undist

# hamiltonian quaternion convention
def qvec2rotmat(qvec):  # TODO: double check this
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])


if __name__ == "__main__":
	args = parse_args()
	text_undist = ""
	if args.run_colmap:
		run_colmap(args)
	if args.run_undistort:
		text_undist = undistort(args)

	IMAGE_FOLDER = args.images
	TEXT_FOLDER = text_undist
	OUT_PATH = args.out
	print(f"outputting to {OUT_PATH}...")
	with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
		angle_x = math.pi / 2
		for line in f:
			# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
			# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
			# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
			if line[0] == "#":
				continue
			els = line.split(" ")
			w = float(els[2])
			h = float(els[3])
			fl_x = float(els[4])
			fl_y = float(els[4])
			k1 = 0
			k2 = 0
			k3 = 0
			k4 = 0
			p1 = 0
			p2 = 0
			cx = w / 2
			cy = h / 2
			is_fisheye = False
			if els[1] == "SIMPLE_PINHOLE":
				cx = float(els[5])
				cy = float(els[6])
			elif els[1] == "PINHOLE":
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
			# elif els[1] == "OPENCV":
			# 	fl_y = float(els[5])
			# 	cx = float(els[6])
			# 	cy = float(els[7])
			# 	k1 = float(els[8])
			# 	k2 = float(els[9])
			# 	p1 = float(els[10])
			# 	p2 = float(els[11])
			else:
				print("Unknown camera model ", els[1])
			# fl = 0.5 * w / tan(0.5 * angle_x);
			angle_x = math.atan(w / (fl_x * 2)) * 2
			angle_y = math.atan(h / (fl_y * 2)) * 2
			fovx = angle_x * 180 / math.pi
			fovy = angle_y * 180 / math.pi

	print(
		f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

	with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
		i = 0
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
		out = {
			"camera_angle_x": angle_x,
			"camera_angle_y": angle_y,
			"fl_x": fl_x,
			"fl_y": fl_y,
			"k1": k1,
			"k2": k2,
			"k3": k3,
			"k4": k4,
			"p1": p1,
			"p2": p2,
			"is_fisheye": is_fisheye,
			"cx": cx,
			"cy": cy,
			"w": w,
			"h": h,
			"frames": [],
		}
		
		for line in f:
			line = line.strip()
			if line[0] == "#":
				continue
			i = i + 1
			if  i % 2 == 1:
				elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
				#name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
				# why is this requireing a relitive path while using ^
				image_rel = os.path.relpath(IMAGE_FOLDER)
				name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(qvec) # hamiltonian # w,x,y,z
				#R_test = Rotation.from_quat([qvec[1],qvec[2], qvec[3], qvec[0]]) #GtoC   #x,y,z,w
				#print("R", R)
				#print("R_test", R_test.as_matrix())
				t = tvec.reshape([3,1]) #->GinC
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0) #R_WtoC
				c2w = np.linalg.inv(m) #R_CtoW, p_WinC
				# in our language world to cam
				frame = {"file_path":name,"transform_matrix": c2w}
				out["frames"].append(frame)
				
	nframes = len(out["frames"])
	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
	print("done")