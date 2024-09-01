import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.image_utils import min_max_norm
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import pickle
from pdb import set_trace as stx


class ConeGeometry(object):
    def __init__(self, data):

        scale = 1.0

        self.DSD = data["DSD"]/scale 
        self.DSO = data["DSO"]/scale

        self.nDetector = np.array(data["nDetector"])  
        self.dDetector = np.array(data["dDetector"])/scale  
        self.sDetector = self.nDetector * self.dDetector  
        
        self.nVoxel = np.array(data["nVoxel"])  
        self.dVoxel = np.array(data["dVoxel"])/scale  
        self.sVoxel = self.nVoxel * self.dVoxel 

        self.offOrigin = np.array(data["offOrigin"])/scale
        self.offDetector = np.array(data["offDetector"])/scale 

        self.accuracy = data["accuracy"]
        self.mode = data["mode"]
        self.filter = data["filter"]




def get_voxels(geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2  

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel



class CameraInfo(NamedTuple):
    uid: int            
    R: np.array         
    T: np.array         
    FovY: np.array      
    FovX: np.array      
    image: np.array     
    image_path: str     
    image_name: str     
    width: int          
    height: int         


class CameraInfo_Xray(NamedTuple):
    uid: int            
    R: np.array         
    T: np.array         
    FovY: np.array      
    FovX: np.array      
    image: np.array     
    image_name: str     
    width: int          
    height: int         
    angle: float        


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud    
    train_cameras: list             
    test_cameras: list              
    add_cameras: list               
    nerf_normalization: dict       
    ply_path: str                   



def getNerfppNorm(cam_info):

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)                           
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)    
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)



def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        fovx = contents["camera_angle_x"]   

        frames = contents["frames"]        
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1


            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  
            T = w2c[:3, 3]


            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem    
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0

            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos



def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3    
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info




def angle2pose(DSO, angle):
    phi1 = -np.pi / 2
    R1 = np.array([[1.0, 0.0, 0.0],
                [0.0, np.cos(phi1), -np.sin(phi1)],
                [0.0, np.sin(phi1), np.cos(phi1)]])

    phi2 = np.pi / 2
    R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                [np.sin(phi2), np.cos(phi2), 0.0],
                [0.0, 0.0, 1.0]])

    R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0]])
    rot = np.dot(np.dot(R3, R2), R1)

    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0]) 
    T = np.eye(4)
    T[:-1, :-1] = rot
    T[:-1, -1] = trans  
    return T



def Xray_readCamerasFromTransforms(path, type = 'train'):

    cam_infos = []
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    geometry = ConeGeometry(data)

    projs = data[type]["projections"]
    angles = data[type]["angles"]
    h, w = projs[0].shape
    fovx = focal2fov(geometry.DSD, w)
    

    for idx, image_arr in enumerate(projs):
        c2w = angle2pose(geometry.DSO,angles[idx])
        image_name = str(idx)

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3]) 
        T = w2c[:3, 3]


        image = image_arr
        angle = angles[idx]

        fovy = focal2fov(geometry.DSD, h)
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo_Xray(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_name=image_name, width=image.shape[0], height=image.shape[1], angle=angle))
            
    return cam_infos



def Xray_readCamerasFromTransforms_addtional(path, add_num = 50):

    cam_infos = []
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    geometry = ConeGeometry(data)

    type = 'train'
    h, w = data[type]["projections"][0].shape
    projs = np.zeros((add_num, h, w))
    angles = np.random.uniform(0, np.pi, add_num)
    fovx = focal2fov(geometry.DSD, w)
    

    for idx, image_arr in enumerate(projs):

        c2w = angle2pose(geometry.DSO,angles[idx])
        image_name = str(idx)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3]) 
        T = w2c[:3, 3]

        image = image_arr
        angle = angles[idx]

        fovy = focal2fov(geometry.DSD, h)
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo_Xray(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_name=image_name, width=image.shape[0], height=image.shape[1], angle=angle))
            
    return cam_infos



def Xray_readNerfSyntheticInfo(path, eval, cube_pcd_init = True, interval = 2, add_num = 50, train_num = 50):
    print("Reading Training Transforms")
    train_cam_infos = Xray_readCamerasFromTransforms(path, type = "train")
    print("Reading Test Transforms")
    test_cam_infos = Xray_readCamerasFromTransforms(path, type = "val")
    print("creating additional camera poses")
    add_cam_infos = Xray_readCamerasFromTransforms_addtional(path, add_num = add_num)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)


    data_dir = os.path.dirname(path)
    ply_path = os.path.join(data_dir, "points3d.ply")

    if cube_pcd_init:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        geometry = ConeGeometry(data)
        pt_positions = get_voxels(geometry)
        image_3d = data["image"]
        s1, s2, s3, _ = pt_positions.shape
        sampled_positions = pt_positions[::interval,::interval,::interval]
        
        xyz = sampled_positions.reshape(-1,3)
        num_pts = xyz.shape[0]
        print(f"Generating point cloud from uniform cube ({num_pts})")
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    elif not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)    
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[:train_num],
                           test_cameras=test_cam_infos,
                           add_cameras=add_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, pt_positions, image_3d




sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Xray": Xray_readNerfSyntheticInfo
}