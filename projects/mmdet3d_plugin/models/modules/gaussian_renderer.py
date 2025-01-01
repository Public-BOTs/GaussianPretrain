import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from mmcv.runner import force_fp32, auto_fp16


def render(viewmatrix, projmatrix, cam_pos, pts_xyz, pts_rgb, rotations, scales, opacity, height, width, fovx, fovy,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color=[0, 0, 0]
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    # tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=3,
        campos=cam_pos,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_img_rgb, rendered_img_radii, rendered_img_depth= rasterizer(
    #     means3D=pts_xyz,
    #     means2D=screenspace_points,
    #     shs=None,
    #     colors_precomp=pts_rgb,
    #     opacities=opacity,
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=None)

    rendered_img_rgb, rendered_img_radii, rendered_img_depth= rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_img_rgb, rendered_img_radii > 0, rendered_img_depth