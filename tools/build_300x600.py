"""300x600 da master 320x480 (forniti a 1024x1536).
Coordinate di layout in unita' di design 1024x2048 (stessa densita' del master);
F = scala di render (1.0 -> 1024x2048 HD; 300/1024 -> 300x600 esatto, un solo ricampionamento).
Strategia: fondo = ambient del master (oggetti/testi rimossi) scalato a copertura
dell'altezza; soggetto e CTA dal master fusi in dominio gradiente; testi e loghi
come asset estratti dai pixel originali, mai ridisegnati."""
import sys, math, lib
from lib import *
W,H=1024,2048
SA=H/1536  # ambient a copertura altezza (1365x2048, rifilo centrale)

def ambient_canvas(amb, y_off=0):
    Wf=int(round(W*lib.F)); Hf=int(round(H*lib.F))
    a=resize(amb,SA*lib.F); ah,aw=a.shape[:2]; xa=(aw-Wf)//2
    return a[y_off:y_off+Hf, xa:xa+Wf].copy()

def clone(dst, src, s, x, y, poly, mode=cv2.NORMAL_CLONE, feather=70):
    """porzione `poly` (coord. sorgente) di `src` a scala s, angolo sorgente (0,0) in (x,y) di design."""
    F=lib.F; Hf,Wf=dst.shape[:2]
    P=300
    sp=cv2.copyMakeBorder(src,P,P,P,P,cv2.BORDER_REFLECT)
    p=resize(sp,s*F); X=int(round((x-P*s)*F)); Y=int(round((y-P*s)*F))
    srcimg=np.zeros((Hf,Wf,3),np.uint8); ph,pw=p.shape[:2]
    cx0=max(0,X); cy0=max(0,Y); cx1=min(Wf,X+pw); cy1=min(Hf,Y+ph)
    srcimg[cy0:cy1,cx0:cx1]=np.clip(p[cy0-Y:cy1-Y,cx0-X:cx1-X],0,255).astype(np.uint8)
    m=np.zeros((Hf,Wf),np.uint8)
    pts=np.array([[(x+px*s)*F,(y+py*s)*F] for px,py in poly],np.int32)
    cv2.fillPoly(m,[pts],255)
    m[:2]=0; m[-2:]=0; m[:,:2]=0; m[:,-2:]=0
    ys,xs=np.where(m>0)
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    c=((x0+x1+1)//2,(y0+y1+1)//2)
    d=np.clip(dst,0,255).astype(np.uint8)
    out=cv2.seamlessClone(srcimg,d,m,c,mode).astype(np.float32)
    # dissolvenza della texture verso il fondo sui bordi interni (non sui bordi del canvas)
    mm=np.zeros((Hf,Wf),np.uint8); cv2.fillPoly(mm,[pts],1)
    q=cv2.copyMakeBorder(mm,8,8,8,8,cv2.BORDER_REPLICATE)
    dist=cv2.distanceTransform(q,cv2.DIST_L2,5)[8:-8,8:-8]
    a=np.clip(dist/(feather*F),0,1); a=(a*a*(3-2*a))[...,None]
    return dst*(1-a)+out*a

def ellipse(cx,cy,rx,ry,n=48):
    return [(cx+rx*math.cos(t/n*2*math.pi), cy+ry*math.sin(t/n*2*math.pi)) for t in range(n)]

def headline_rows(cv,E,B,rows,k,y,gap_x=20,gap_y=-5):
    for row in rows:
        ws=[(B[key][2]-B[key][0])*k for key in row]; hs=[(B[key][3]-B[key][1])*k for key in row]
        x=(W-(sum(ws)+gap_x*(len(row)-1)))/2
        for key,ww in zip(row,ws): paste(cv,*E[key],x,y,k); x+=ww+gap_x
        y+=max(hs)+gap_y
    return y

def stack(cv,E,B,keys,gaps,k,y):
    for key,g in zip(keys,gaps+[0]):
        ww=(B[key][2]-B[key][0])*k; hh=(B[key][3]-B[key][1])*k
        paste(cv,*E[key],(W-ww)/2,y,k); y+=hh+g
    return y

def cta_layer(cv, src, amb, box, y, k=1.0):
    """CTA dal master, alpha = differenza dal fondo ricostruito (porta bottone + glow)."""
    x0,y0,x1,y1=box
    d=np.abs(src-amb).max(axis=2)
    a=np.clip((d-6)/30,0,1)
    m=np.zeros(src.shape[:2],np.float32); m[y0:y1,x0:x1]=1
    m=cv2.GaussianBlur(m,(0,0),6)
    a=cv2.GaussianBlur(a,(0,0),1.5)*m
    ww=(x1-x0)*k
    paste(cv,src[y0:y1,x0:x1].copy(),a[y0:y1,x0:x1],(W-ww)/2,y,k)
    return y+(y1-y0)*k

def slide1():
    src=load(1)
    B={'scopri':(160,70,865,195),'bonus':(205,192,840,455),'casino':(195,455,850,710),'oggi':(320,712,705,845)}
    CTA=(95,1300,930,1510)
    M={k:textmask(src,[b]) for k,b in B.items()}
    allm=np.zeros_like(M['scopri'])
    for m in M.values(): allm|=m
    pl=plate(src,allm)
    M={k:textmask(src,[b],dil=5) for k,b in B.items()}
    amb=ambient_soft(pl,[(200,820,830,1290),(110,1310,915,1500)])
    cv=ambient_canvas(amb)
    s=1.30; cy=1265  # centro corona in design
    cv=clone(cv,pl,s,512-512*s,cy-1060*s,ellipse(512,1060,370,265))
    E={key:element_matte(src,pl,M[key],B[key]) for key in B}
    y=headline_rows(cv,E,B,[['scopri'],['bonus'],['casino'],['oggi']],1.0,130,gap_y=-3)
    yc=cta_layer(cv,src,amb,CTA,2048-130-210)
    return cv,(y,yc)

def slide5():
    src=load(5)
    B={'confronta':(262,62,798,195),'imigliori':(242,195,808,327),'bonus':(162,328,872,560),'casino':(148,560,897,788)}
    CTA=(115,1300,910,1500)
    M={k:textmask(src,[b]) for k,b in B.items()}
    allm=np.zeros_like(M['bonus'])
    for m in M.values(): allm|=m
    pl=plate(src,allm)
    M={k:textmask(src,[b],dil=5) for k,b in B.items()}
    amb=ambient_soft(pl,[(320,760,710,1170),(20,1160,1004,1320),(130,1310,895,1500)])
    cv=ambient_canvas(amb)
    s=1.40  # punta del cristallo sotto CASINO, come nel master
    top=[(x,790-45*math.exp(-((x-512)/120)**2)) for x in range(4,1021,34)]
    arc=[(x,1195+105*math.sqrt(max(0,1-((x-512)/540)**2))) for x in range(1020,3,-34)]
    cv=clone(cv,pl,s,512-512*s,845-790*s,top+arc)
    E={key:element_matte(src,pl,M[key],B[key]) for key in B}
    y=headline_rows(cv,E,B,[['confronta'],['imigliori'],['bonus'],['casino']],1.0,130,gap_y=0)
    yc=cta_layer(cv,src,amb,CTA,2048-130-200)
    return cv,(y,yc)

def brand(i,B,k,top,gaps,s,obj_poly,obj_anchor,logo_blue=False,amb_boxes=None,amt_lt=95):
    """obj_anchor=(y_sorgente, y_design): la riga y_sorgente del master finisce a y_design."""
    src=load(i)
    M={key:textmask(src,[B[key]],blue=(logo_blue and key=='logo')) for key in B}
    allm=rectmask(src.shape,[B['logo']])|M['fino']|M['amt']
    pl=plate(src,allm)
    amb=ambient_soft(pl,amb_boxes) if amb_boxes else pl
    cv=ambient_canvas(amb)
    ys,yd=obj_anchor
    cv=clone(cv,pl,s,512-512*s,yd-ys*s,obj_poly)
    M['amt']=textmask(src,[B['amt']],lt=amt_lt,dil=10 if amt_lt<=95 else 6)  # soglia alta: niente bagliore del master nei vuoti delle cifre
    E={key:element(src,M[key],B[key]) for key in B}
    y=stack(cv,E,B,['logo','fino','amt'],gaps,k,top)
    return cv,(y,)

def slide2():
    B={'logo':(120,58,895,288),'fino':(355,312,665,388),'amt':(68,393,978,688)}
    return brand(2,B,1.0,150,[30,8],1.12,[(x,700-60*math.exp(-((x-512)/110)**2)) for x in range(4,1021,34)]+[(1020,1532),(4,1532)],
                 (1536,2048),logo_blue=True,amb_boxes=[(60,660,964,1536)],amt_lt=140)
def slide3():
    B={'logo':(195,80,840,285),'fino':(380,340,655,408),'amt':(155,415,875,672)}
    return brand(3,B,1.1,160,[40,10],1.25,[(4,720),(1020,720),(1020,1532),(4,1532)],(1450,1900),
                 amb_boxes=[(250,780,780,1500)])
def slide4():
    B={'logo':(165,92,855,232),'fino':(395,282,625,348),'amt':(220,348,825,568)}
    return brand(4,B,1.25,170,[50,10],1.25,[(4,640),(1020,640),(1020,1532),(4,1532)],(1445,1880),
                 amb_boxes=[(240,660,800,1480)])

if __name__=='__main__':
    lib.setF(float(sys.argv[1])); out=sys.argv[2]
    only=[int(a) for a in sys.argv[3:]] or [1,2,3,4,5]
    names={1:'01_scopri-bonus-casino',2:'02_staryes-1050',3:'03_netwin-500',4:'04_pokerstars-310',5:'05_confronta-bonus-casino'}
    fns={1:slide1,2:slide2,3:slide3,4:slide4,5:slide5}
    for i in only:
        cv,info=fns[i]()
        save(cv,f'{out}/{names[i]}.png'); print(i,cv.shape,info)
