from lib import *
src=load(5)
B={'confronta':(262,62,798,195),'imigliori':(242,195,808,327),'bonus':(162,328,872,560),'casino':(148,560,897,788)}
M={k:textmask(src,[b]) for k,b in B.items()}
allm=np.zeros_like(M['bonus'])
for m in M.values(): allm|=m
pl=plate(src,allm); cv2.imwrite('pl5.png',pl.astype(np.uint8))
amb=ambient(pl,[(300,740,730,1180),(10,1150,1014,1330),(120,1300,905,1510)])
W,H=1200,1000; s=0.66; dy=-15
cv=compose_bg(pl,amb,W,H,1.1719,-560,s,(W-1024*s)/2,dy,[(150,500,875,1150),(140,1150,885,1330),(125,1300,900,1510)],f=32)
k=0.715
def w(key): x0,y0,x1,y1=B[key]; return (x1-x0)*k,(y1-y0)*k
E={key:element(src,M[key],B[key]) for key in B}
w1,h1=w('confronta'); w2,h2=w('imigliori'); w3,h3=w('bonus'); w4,h4=w('casino')
g=22
tot=max(h1,h2)+max(h3,h4)
y=(470-tot)/2+20
x=(W-(w1+w2+g))/2; paste(cv,*E['confronta'],x,y,k); paste(cv,*E['imigliori'],x+w1+g,y,k); y+=max(h1,h2)
x=(W-(w3+w4+g))/2; paste(cv,*E['bonus'],x,y,k); paste(cv,*E['casino'],x+w3+g,y,k); print('bottom',y+max(h3,h4), 'crystal top',775*s+dy)
save(cv,'out5.png')
