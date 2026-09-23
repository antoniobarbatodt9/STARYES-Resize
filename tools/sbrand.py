from lib import *
import sys
cfg={
3:dict(B={'logo':(195,80,840,285),'fino':(380,340,655,408),'amt':(155,415,875,672)},s=0.66,dy=40,k=0.78,top=38,gaps=[26,8]),
4:dict(B={'logo':(165,92,855,232),'fino':(395,282,625,348),'amt':(220,348,825,568)},s=0.66,dy=61,k=0.92,top=45,gaps=[34,6]),
}
i=int(sys.argv[1]); c=cfg[i]; B=c['B']
src=load(i)
M={key:textmask(src,[B[key]]) for key in B}
allm=rectmask(src.shape,[B['logo']])|M['fino']|M['amt']
pl=plate(src,allm); cv2.imwrite(f'pl{i}.png',pl.astype(np.uint8))
W,H=1200,1000; s=c['s']; dy=c['dy']
cv=compose_bg(pl,pl,W,H,1.1719,0,s,(W-1024*s)/2,dy,[(70,-300,954,1836)],f=40)
k=c['k']
def w(key): x0,y0,x1,y1=B[key]; return (x1-x0)*k,(y1-y0)*k
E={key:element(src,M[key],B[key]) for key in B}
y=c['top']
for key,gap in zip(['logo','fino','amt'],c['gaps']+[0]):
    ww,hh=w(key); paste(cv,*E[key],(W-ww)/2,y,k); y+=hh+gap
print('bottom',y)
save(cv,f'out{i}.png')
