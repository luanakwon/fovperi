import numpy as np
import math
import torch

def get_inv_FCG_mapping(nr,p0,pmax,c_gaze, dtype=np.int32):
    """
    Returns inverse mapping of Foveal Cartesian Geometry.
    This inverse mapping is required for the cv2::remap to perform foveation
    
    Parameters
    nr : Integer
        Number of rings, not including fovea. nr > 0.
    pmax : Integer
        Maximum distance where the last ring should be positioned. Pixel unit. pmax > p0.
    p0 : Integer
        Fovea radius. Pixel Unit. p0 > 0
    c_gaze : (Integer, Integer)
        In (x0, y0) shape. Center point of gaze in src image.
    dtype : str or dtype
        Typecode or data-type to which the returned arrays are cast. default np.int32
        
    Returns
        X, Y : ndarray
        inverse mapping X and Y. arrays are the same size with the foveated image.
    """
    # factor a
    a = math.exp(1/nr*math.log(pmax/p0))
    # Delta p(Xi)
    delta_p = lambda xi: (p0*a**xi)/(p0+xi) # xi = 1,2,...,Nr
    x0, y0 = c_gaze
    # center point of foveated image
    fcx = fcy = p0+nr
    # get meshgrid of x' and y' to use np
    dx = np.arange(0,(p0+nr)*2+1)
    dy = np.arange(0,(p0+nr)*2+1)
    X_out, Y_out = np.meshgrid(dx,dy) # _out is equal to ' in Table 1.
    # inverse mapping algorithm as in Table 1.
    X_out = X_out - fcx
    Y_out = Y_out - fcy
    R = np.max(np.stack((np.abs(X_out),np.abs(Y_out))),axis=0)
    # In the paper, case where r <= p0 is defined separately.
    # However, it is not necessary if XI = 0 for XI < 0
    XI = np.clip(R-p0,0,nr)
    X = np.floor(X_out*delta_p(XI)+x0).astype(dtype)
    Y = np.floor(Y_out*delta_p(XI)+y0).astype(dtype)
    return X,Y

def get_FCG_revertFunc(nr,p0,pmax):
    """
    Returns the function to undo the FCG for a given coord value
    To revert the radius info, calculate the D conversion
    and use it as a multiplier
    """
    a = math.exp(1/nr*math.log(pmax/p0))
    delta_p = lambda xi: (p0*a**xi)/(p0+xi) # xi = 1,2,...,Nr

    def revertFunc(x):
        """
        x: Tensor of shape (B,2), in pixel coord (y,x),
        x is assumed to have origin at the center of fovea
        returns
            y, x in pixel coord, origin at the center of fovea.
            apply scale then translate to get the original image coord
        """
        xi = x.clip(torch.abs(x)-p0,0,nr)
        x = x*delta_p(xi)
        return x
    
    return revertFunc

def batch_remap(src, base, scale, translate, borderValue=0):
    """
    cv2.remap like operation using tensor indexing.
    Remap implementation for batch operation
    
    Parameters
    src : torch.tensor
        source batch. Expected to be (B,C,H,W)
    base : torch.tensor
        Tuple of (X_mapping, Y_mapping). mappings must be numeric
    scale : Tensor of shape (B,2). 
        scale(zoom) coefficient for each image in batch. x, y order
    translate : Tensor of shape (B,2)
        x y translate for each image in batch. 
    borderValue : Integer
        border fill. TODO) also allow per channel value
    """
    b,c,h,w = src.shape
    out = []
    for maps, sc, tr, dim in zip(base,scale.T,translate.T,(w,h)):
        # apply scale and translate
        maps = (maps.repeat(b,1,1) * sc.view(b,1,1)
                + tr.view(b,1,1)).to(torch.int32)
        off_border = torch.logical_or(maps<0,maps>=dim)
        # since python allows negative indexing, use -1 as a off_border flag
        out.append(maps*(~off_border) - 1*(off_border))
    
    X, Y = out
    b_idx = torch.arange(b).view(b,1,1,1)
    c_idx = torch.arange(c).view(c,1,1)
    # apply mapping to the src
    dst = src[b_idx,c_idx,Y.unsqueeze(1),X.unsqueeze(1)]
    # fill border
    off_border = torch.logical_or(X < 0, Y < 0).unsqueeze(1)
    dst = dst*(~off_border) + borderValue*off_border
    
    return dst
