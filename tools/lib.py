import cv2, numpy as np
from PIL import Image
F=1.0
def setF(v):
    global F; F=v
import glob, os
def load(i): return cv2.imread(sorted(glob.glob(os.path.join(os.path.dirname(__file__),'..','master','320x480',f'0{i}_*.webp')))[0]).astype(np.float32)
def textmask(src, boxes, dil=10, blue=False, lt=95):
    m=np.zeros(src.shape[:2],np.uint8)
    b,g,r=[src[...,k] for k in range(3)]
    lum=0.299*r+0.587*g+0.114*b
    fg=((lum>lt)|((r>110)&(r>b+45)))
    if blue: fg|=((b>120)&(b>r+60))
    fg=fg.astype(np.uint8)
    for (x0,y0,x1,y1) in boxes:
        sub=fg[y0:y1,x0:x1].copy()
        sub=cv2.morphologyEx(sub,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        sub=cv2.dilate(sub,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)))
        m[y0:y1,x0:x1]|=sub
    return m
def plate(src, mask):
    s=src.astype(np.uint8)
    sm=cv2.resize(s,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    mm=cv2.resize(mask,(sm.shape[1],sm.shape[0]),interpolation=cv2.INTER_NEAREST)
    mm=cv2.dilate(mm,np.ones((5,5),np.uint8))
    ip=cv2.inpaint(sm,mm*255,12,cv2.INPAINT_TELEA)
    ip=cv2.GaussianBlur(ip,(0,0),3)
    ip=cv2.resize(ip,(s.shape[1],s.shape[0]),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    a=cv2.GaussianBlur(cv2.dilate(mask,np.ones((9,9),np.uint8)).astype(np.float32),(0,0),4)[...,None]
    return src*(1-a)+ip*a
def element(src, mask, box, feather=2.0):
    x0,y0,x1,y1=box
    a=cv2.GaussianBlur(mask.astype(np.float32),(0,0),feather)[y0:y1,x0:x1]
    return src[y0:y1,x0:x1].copy(), a
def resize(img, s):
    h,w=img.shape[:2]; nw,nh=max(1,int(round(w*s))),max(1,int(round(h*s)))
    if img.ndim==2:
        return np.array(Image.fromarray(img.astype(np.float32),'F').resize((nw,nh),Image.LANCZOS))
    return np.dstack([np.array(Image.fromarray(img[...,c].astype(np.float32),'F').resize((nw,nh),Image.LANCZOS)) for c in range(3)])
def paste(canvas, rgb, a, x, y, s):
    x*=F; y*=F; s*=F
    r=resize(rgb,s); aa=np.clip(resize(a,s),0,1)[...,None]
    h,w=r.shape[:2]; x=int(round(x)); y=int(round(y))
    canvas[y:y+h,x:x+w]=canvas[y:y+h,x:x+w]*(1-aa)+r*aa
    return (x,y,w,h)
def bg_canvas(pl, W, H, s, dy):
    """plate scaled by s, centered horizontally, offset dy; sides filled by mirror reflection."""
    p=resize(pl,s); ph,pw=p.shape[:2]
    x0=(W-pw)//2
    # reflect-pad horizontally
    padl=x0; padr=W-pw-x0
    big=cv2.copyMakeBorder(p,0,0,padl,padr,cv2.BORDER_REFLECT)
    # vertical: crop/pad
    out=np.zeros((H,W,3),np.float32)
    ys=max(0,-dy); yd=max(0,dy); h=min(ph-ys,H-yd)
    out[yd:yd+h]=big[ys:ys+h]
    if yd>0: out[:yd]=big[ys:ys+1]  # shouldn't happen much
    return out, x0
def save(canvas, path):
    c=np.clip(np.round(canvas),0,255).astype(np.uint8)
    cv2.imwrite(path,c)
def rectmask(shape, boxes):
    m=np.zeros(shape[:2],np.uint8)
    for x0,y0,x1,y1 in boxes: m[y0:y1,x0:x1]=1
    return m
def ambient(pl, objboxes, scale=0.25, blur=6):
    s=pl.astype(np.uint8)
    sm=cv2.resize(s,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    m=rectmask(pl.shape,objboxes)
    mm=cv2.resize(m,(sm.shape[1],sm.shape[0]),interpolation=cv2.INTER_NEAREST)
    ip=cv2.inpaint(sm,mm*255,20,cv2.INPAINT_TELEA)
    ip=cv2.GaussianBlur(ip,(0,0),blur)
    ip=cv2.resize(ip,(s.shape[1],s.shape[0]),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    a=np.maximum(m.astype(np.float32),cv2.GaussianBlur(m.astype(np.float32),(0,0),25))[...,None]
    return pl*(1-a)+ip*a
def feathered_rect(h,w,box,f):
    m=np.zeros((h,w),np.float32); x0,y0,x1,y1=[int(v) for v in box]
    m[max(0,y0):y1,max(0,x0):x1]=1
    return cv2.GaussianBlur(m,(0,0),f)
def compose_bg(pl, amb, W,H, s_amb, y_amb, s_obj, x_obj, y_obj, objbox, f=40):
    W=int(round(W*F)); H=int(round(H*F)); s_amb*=F; y_amb=int(round(y_amb*F)); s_obj*=F; x_obj*=F; y_obj*=F; f*=F
    a=resize(amb,s_amb); ah,aw=a.shape[:2]
    xa=(W-aw)//2
    cv=np.zeros((H,W,3),np.float32)
    cv[:]=a[-y_amb:-y_amb+H, -xa:-xa+W] if xa<=0 else 0
    P=300
    plp=cv2.copyMakeBorder(pl,P,P,P,P,cv2.BORDER_REFLECT)
    p=resize(plp,s_obj); ph,pw=p.shape[:2]
    x_obj-=P*s_obj; y_obj-=P*s_obj
    boxes=objbox if isinstance(objbox,list) else [objbox]
    boxes=[(a+P,b+P,c+P,d+P) for a,b,c,d in boxes]
    m=np.zeros((ph,pw),np.float32)
    for x0,y0,x1,y1 in boxes: m[max(0,int(y0*s_obj)):int(y1*s_obj),max(0,int(x0*s_obj)):int(x1*s_obj)]=1
    m=cv2.GaussianBlur(m,(0,0),f)[...,None]
    # place p on canvas with clipping
    X=int(x_obj); Y=int(y_obj)
    cx0=max(0,X); cy0=max(0,Y); cx1=min(W,X+pw); cy1=min(H,Y+ph)
    pp=p[cy0-Y:cy1-Y, cx0-X:cx1-X]; mm=m[cy0-Y:cy1-Y, cx0-X:cx1-X]
    cv[cy0:cy1,cx0:cx1]=cv[cy0:cy1,cx0:cx1]*(1-mm)+pp*mm
    return cv
