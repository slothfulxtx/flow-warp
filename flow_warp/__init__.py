import torch
import torch.nn.functional as F
from . import _C

class _ForwardWarp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, flow, mode) -> torch.Tensor:
        
        warped_x = _C.forward_warp(x, flow, mode)
        ctx.save_for_backward(x, flow)
        ctx.mode = mode
        return warped_x

    @staticmethod
    def backward(ctx, grad_warped_x):
        mode = ctx.mode
        x, flow = ctx.saved_tensors
        grad_x, grad_flow = _C.forward_warp_backward(grad_warped_x, x, flow, mode)
        return (grad_x, grad_flow, None)
        

def forward_warp(x:torch.Tensor, flow: torch.Tensor, mode: str='bilinear'):

    """
        x: feature map with shape [B, C, H, W]
        flow: optical flow with shape [B, H, W, 2], range from [-W, -H] to [W, H]
        mode: bilinear or nearest
    """

    assert x.ndim == 4 and flow.ndim == 4
    B, C, H, W = x.shape
    assert flow.shape == (B, H, W, 2)
    assert mode in ['bilinear', 'nearest']
    assert(torch.isnan(flow).long().sum() == 0)
    assert(torch.isinf(flow).long().sum() == 0)

    x = x.contiguous()
    flow = flow.contiguous()
    mode_num = 0 if mode == 'bilinear' else 1
    return _ForwardWarp.apply(x, flow, mode_num)
        
def backward_warp(x:torch.Tensor, flow: torch.Tensor, mode: str='bilinear'):

    """
        x: feature map with shape [B, C, H, W]
        flow: optical flow with shape [B, H, W, 2], range from [-W, -H] to [W, H]
        mode: bilinear or nearest
    
    """

    assert x.ndim == 4 and flow.ndim == 4
    B, C, H, W = x.shape
    assert flow.shape == (B, H, W, 2)
    assert mode in ['bilinear', 'nearest']
    assert(torch.isnan(flow).long().sum() == 0)
    assert(torch.isinf(flow).long().sum() == 0)

    x = x.contiguous()
    flow = flow.contiguous()
    
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    # H, W
    xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1)
    yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), -1).float().to(x.device)
    # B, H, W, 2
    vgrid = grid + flow
    
    vgrid = torch.cat([
        2.0 * vgrid[:, :, :, 0:1] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, :, :, 1:2] / max(H - 1, 1) - 1.0,
    ], dim=-1)
   
    output = F.grid_sample(x, vgrid, align_corners=True, mode=mode)
    
    return output


