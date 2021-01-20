import argparse
import os
import time

import imageio
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from lieutils import SE3, SE3Exp
from tqdm import tqdm
import matplotlib.pyplot as plt

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_blender_data_inerf, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, plot_grad_flow)

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    img = img.detach().cpu().numpy().astype(np.uint8)
    return np.expand_dims(img, 0)


def load_dataset(cfg):
    images, ref_poses, render_poses, hwf = None, None, None, None
    if cfg.type.lower() == "blender":
        # Load blender dataset
        images, ref_poses, render_poses, hwf, _ = load_blender_data_inerf(
            cfg.basedir,
            half_res=cfg.half_res,
            testskip=cfg.testskip,
        )
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.type.lower() == "llff":
        # Load LLFF dataset
        images, ref_poses, _, render_poses, _ = load_llff_data(
            cfg.basedir, factor=cfg.downsample_factor,
        )
        hwf = ref_poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    return images, ref_poses, render_poses, hwf


def load_embeddings(cfg):
    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.num_encoding_fn_xyz,
        include_input=cfg.include_input_xyz,
        log_sampling=cfg.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.num_encoding_fn_dir,
            include_input=cfg.include_input_dir,
            log_sampling=cfg.log_sampling_dir,
        )

    return encode_position_fn, encode_direction_fn


def load_models(cfg):
    model_coarse = getattr(models, cfg.coarse.type)(
        num_encoding_fn_xyz=cfg.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.coarse.include_input_xyz,
        include_input_dir=cfg.coarse.include_input_dir,
        use_viewdirs=cfg.coarse.use_viewdirs,
    )

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg, "fine"):
        model_fine = getattr(models, cfg.fine.type)(
            num_encoding_fn_xyz=cfg.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.fine.include_input_xyz,
            include_input_dir=cfg.fine.include_input_dir,
            use_viewdirs=cfg.fine.use_viewdirs,
        )
    return model_coarse, model_fine


def load_checkpoint(cfg, model_coarse, model_fine):
    checkpoint = torch.load(cfg.checkpoint)

    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    return checkpoint


def main(cfg, configargs):

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, ref_poses, render_poses, hwf = load_dataset(cfg.dataset)

    # Obtain the position and direction embedding functions (length, and skip)
    encode_position_fn, encode_direction_fn = load_embeddings(cfg.models.coarse)

    # Initialize a coarse resolution model.
    model_coarse, model_fine = load_models(cfg.models)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_coarse.to(device)
    if model_fine:
        model_fine.to(device)

    # Load an existing checkpoint
    checkpoint = load_checkpoint(configargs, model_coarse, model_fine)
    H, W, focal = hwf[0], hwf[1], hwf[2]
    if "height" in checkpoint.keys():
        H = checkpoint["height"]
    if "width" in checkpoint.keys():
        W = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        focal = checkpoint["focal_length"]

    ref_poses = ref_poses.float().to(device)
    render_poses = render_poses.float().to(device)

    # Setup logging
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # Create directory to save images to.
    os.makedirs(configargs.savedir, exist_ok=True)
    if configargs.save_disparity_image:
        os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)

    # iNERF Training loop
    times_per_image = []
    for i, cam_pose in enumerate(tqdm(render_poses)):
        start = time.time()

        ref_pose = ref_poses[i, :]
        tform_cam2ref = torch.matmul(torch.inverse(ref_pose), cam_pose)
        rgb = None, None
        disp = None, None

        # Optimization variable
        delta_pose = torch.normal(0.0, torch.empty(1, 6).fill_(1e-6)).to(device)
        delta_pose.requires_grad = True
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [delta_pose], lr=cfg.optimizer.lr
        )

        model_coarse.eval()
        if model_fine:
            model_fine.eval()

        ref_ray_origins, ref_ray_directions = get_ray_bundle(H, W, focal, ref_pose)
        ref_rgb_fine, ref_disp_fine = None, None
        with torch.no_grad():
            _, _, _, ref_rgb_fine, ref_disp_fine, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ref_ray_origins,
                ref_ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        writer.add_image(f"Reference RGB {i}", cast_to_image(ref_rgb_fine[..., :3]))
        writer.add_image(f"Reference disparity {i}", cast_to_disparity_image(ref_disp_fine))

        model_coarse.train()
        if model_fine:
            model_fine.train()
        cam_pose_initial = cam_pose
        for j in tqdm(range(cfg.experiment.train_iters)):

            # 1. Compose the newly updated delta_pose with initial camera_pose estimate
            cam_pose = SE3Exp(delta_pose).squeeze() @ cam_pose_initial

            # 2. Obtain rays at new pose
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, cam_pose)

            # 3. Choose a few interesting rays (within cuda limit) from the whole image
            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            coords = coords.reshape((-1, 2))
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            )
            select_inds = coords[select_inds]
            ray_origins_sample = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions_sample = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            ref_rgb_fine_sample = ref_rgb_fine[select_inds[:, 0], select_inds[:, 1], :]

            # 4. Forward pass via NeRF to get rendered image
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins_sample,
                ray_directions_sample,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )


            # 5. Backward pass via NeRF over delta_pose to get gradient
            coarse_loss = torch.nn.functional.mse_loss(rgb_coarse[..., :3], ref_rgb_fine_sample[..., :3])
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = torch.nn.functional.mse_loss(rgb_fine[..., :3], ref_rgb_fine_sample[..., :3])
            loss = 0.0
            loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
            loss.backward()

            gradient_params = [("delta", delta_pose)]
            gradient_params += model_coarse.named_parameters()
            if model_fine:
                gradient_params += model_fine.named_parameters()
            plot_grad_flow(gradient_params)
            # 6. Metrics
            psnr = mse2psnr(loss.item())
            tform_cam2ref = torch.matmul(torch.inverse(ref_pose), cam_pose)
            pose_error = SE3.Log(tform_cam2ref)
            pose_error = torch.norm(pose_error, 2)

            # 7. Descent over gradient
            optimizer.step()
            optimizer.zero_grad()

            # 8. Learning rate updates
            num_decay_steps = cfg.scheduler.lr_decay
            lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_new

            writer.add_scalar(f"train_{i}/loss", loss.item(), j)
            writer.add_scalar(f"train_{i}/coarse_loss", coarse_loss.item(), j)
            writer.add_scalar(f"train_{i}/psnr", psnr, j)
            writer.add_scalar(f"train_{i}/pose_error", pose_error, j)
            if rgb_fine is not None:
                writer.add_scalar(f"train_{i}/fine_loss", fine_loss.item(), j)
            if j % cfg.experiment.print_every == 0 or j == cfg.experiment.train_iters - 1:
                tqdm.write(f"[TRAIN] Iter: {j} Loss: {loss.item()} PSNR: {psnr}")
                tqdm.write(f"Current relative pose at iteration {j}: \n {tform_cam2ref} \n {delta_pose}")

            times_per_image.append(time.time() - start)

            if j % cfg.experiment.validate_every == 0 or j == cfg.experiment.train_iters - 1:
                tqdm.write("[VAL] =======> Iter: " + str(j))
                model_coarse.eval()
                if model_fine:
                    model_fine.eval()
                with torch.no_grad():
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )

                rgb = rgb_fine if rgb_fine is not None else rgb_coarse
                os.makedirs(os.path.join(configargs.savedir, f"{i}"), exist_ok=True)
                if configargs.save_disparity_image:
                    disp = disp_fine if disp_fine is not None else disp_coarse
                if configargs.savedir:
                    savefile = os.path.join(configargs.savedir, f"{i}", f"{j:04d}.png")
                    imageio.imwrite(savefile, np.moveaxis(cast_to_image(rgb[..., :3]), [0], [-1]))
                    writer.add_image(f"RGB {i}", cast_to_image(rgb[..., :3]), j)
                if configargs.save_disparity_image:
                    savefile = os.path.join(configargs.savedir, f"{i}", "disparity", f"{j:04d}.png")
                    imageio.imwrite(savefile, np.squeeze(cast_to_disparity_image(disp)))
                    writer.add_image(f"Disparity {i}", cast_to_disparity_image(disp), j)
            if (torch.allclose(tform_cam2ref, torch.eye(4).to(device))):
                break

        savefileplt = os.path.join(configargs.savedir, f"gradient_flow{i:04d}.png")
        plt.savefig(savefileplt)
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    main(cfg, configargs)
