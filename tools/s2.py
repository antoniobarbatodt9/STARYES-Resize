from lib import *
src=load(2)
B={'logo':(120,58,895,288),'fino':(355,312,665,388),'amt':(68,393,978,688)}
M={'logo':textmask(src,[B['logo']],blue=True),'fino':textmask(src,[B['fino']]),'amt':textmask(src,[B['amt']])}
allm=rectmask(src.shape,[B['logo']])|M['fino']|M['amt']
pl=plate(src,allm); cv2.imwrite('pl2.png',pl.astype(np.uint8))
amb=pl
cv2.imwrite('amb2.png',amb.astype(np.uint8))
W,H=1200,1000; s=0.66; dy=100
cv=compose_bg(pl,amb,W,H,1.1719,0,s,(W-1024*s)/2,dy,[(70,-300,954,1536)],f=40)
k=0.76
def w(key): x0,y0,x1,y1=B[key]; return (x1-x0)*k,(y1-y0)*k
E={key:element(src,M[key],B[key]) for key in B}
y=40
for key,gap in [('logo',22),('fino',6),('amt',0)]:
    ww,hh=w(key); paste(cv,*E[key],(W-ww)/2,y,k); y+=hh+gap
print('bottom',y,'tip',680*s+dy)
save(cv,'out2.png')
