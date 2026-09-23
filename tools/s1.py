from lib import *
src=load(1)
B={'scopri':(160,70,865,195),'bonus':(205,192,840,455),'casino':(195,455,850,710),'oggi':(320,712,705,845)}
M={k:textmask(src,[b]) for k,b in B.items()}
allm=np.zeros_like(M['scopri'])
for m in M.values(): allm|=m
pl=plate(src,allm)
amb=ambient(pl,[(190,800,840,1300),(100,1300,925,1510)])
cv2.imwrite('amb1.png',amb.astype(np.uint8))
W,H=1200,1000; s=0.66
cv=compose_bg(pl,amb,W,H,1.1719,-560,s,(W-1024*s)/2,-15,[(225,790,800,1300),(105,1310,920,1505)],f=22)
k=0.79
def w(key): x0,y0,x1,y1=B[key]; return (x1-x0)*k,(y1-y0)*k
E={key:element(src,M[key],B[key]) for key in B}
y=45
ws,hs=w('scopri'); paste(cv,*E['scopri'],(W-ws)/2,y,k); y+=hs-4
wb,hb=w('bonus'); wc,hc=w('casino'); gap=20
x=(W-(wb+wc+gap))/2
paste(cv,*E['bonus'],x,y,k); paste(cv,*E['casino'],x+wb+gap,y-1,k); y+=max(hb,hc)-6
wo,ho=w('oggi'); paste(cv,*E['oggi'],(W-wo)/2,y,k); print('bottom',y+ho)
save(cv,'out1.png')
