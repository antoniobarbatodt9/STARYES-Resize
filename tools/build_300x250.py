"""300x250 da master 320x480 (forniti a 1024x1536).
Coordinate di layout in unita' di design 1200x1000; F = scala di render
(0.8 -> 960x800, stessa densita' del master; 0.25 -> 300x250)."""
import sys, lib
from lib import *
W,H=1200,1000
SA=1.1725  # ambient: master scalato a copertura larghezza

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

def slide1():
    src=load(1)
    B={'scopri':(160,70,865,195),'bonus':(205,192,840,455),'casino':(195,455,850,710),'oggi':(320,712,705,845)}
    M={k:textmask(src,[b]) for k,b in B.items()}
    allm=np.zeros_like(M['scopri'])
    for m in M.values(): allm|=m
    pl=plate(src,allm)
    amb=ambient(pl,[(190,800,840,1300),(100,1300,925,1510)])
    s=0.60; dy=945-1492*s   # CTA bottom a 945 (margine come il top)
    cv=compose_bg(pl,amb,W,H,SA,-560,s,(W-1024*s)/2,dy,[(225,790,800,1300),(105,1310,920,1505)],f=22)
    E={key:element(src,M[key],B[key]) for key in B}
    y=headline_rows(cv,E,B,[['scopri'],['bonus','casino'],['oggi']],0.78,52)
    return cv,(y,835*s+dy)

def slide5():
    src=load(5)
    B={'confronta':(262,62,798,195),'imigliori':(242,195,808,327),'bonus':(162,328,872,560),'casino':(148,560,897,788)}
    M={k:textmask(src,[b]) for k,b in B.items()}
    allm=np.zeros_like(M['bonus'])
    for m in M.values(): allm|=m
    pl=plate(src,allm)
    amb=ambient(pl,[(300,740,730,1180),(10,1150,1014,1330),(120,1300,905,1510)])
    s=0.60; dy=945-1484*s
    cv=compose_bg(pl,amb,W,H,SA,-560,s,(W-1024*s)/2,dy,[(150,500,875,1150),(140,1150,885,1330),(125,1300,900,1510)],f=32)
    E={key:element(src,M[key],B[key]) for key in B}
    y=headline_rows(cv,E,B,[['confronta','imigliori'],['bonus','casino']],0.715,70,gap_x=22,gap_y=0)
    return cv,(y,775*s+dy)

def brand(i,B,k,top,gaps,s,obj_bottom_src,obj_bottom_canvas,logo_blue=False,sa=SA,amb_boxes=None):
    src=load(i)
    M={key:textmask(src,[B[key]],blue=(logo_blue and key=='logo')) for key in B}
    allm=rectmask(src.shape,[B['logo']])|M['fino']|M['amt']
    pl=plate(src,allm)
    dy=obj_bottom_canvas-obj_bottom_src*s
    amb=ambient(pl,amb_boxes) if amb_boxes else pl
    cv=compose_bg(pl,amb,W,H,sa,0,s,(W-1024*s)/2,dy,[(70,-300,954,1836)],f=40)
    E={key:element(src,M[key],B[key]) for key in B}
    y=stack(cv,E,B,['logo','fino','amt'],gaps,k,top)
    return cv,(y,dy)

def slide2():
    B={'logo':(120,58,895,288),'fino':(355,312,665,388),'amt':(68,393,978,688)}
    # corona intera: punta cristallo 680 -> base piattaforma 1500
    return brand(2,B,0.68,42,[18,4],0.585,1536,1000,logo_blue=True,amb_boxes=[(225,805,810,900)])  # punte corona tolte dallo sfondo laterale
def slide3():
    B={'logo':(195,80,840,285),'fino':(380,340,655,408),'amt':(155,415,875,672)}
    # podio intero fino a 1450, spazio pavimento sotto come nel master
    return brand(3,B,0.70,42,[22,6],0.60,1450,950)
def slide4():
    B={'logo':(165,92,855,232),'fino':(395,282,625,348),'amt':(220,348,825,568)}
    return brand(4,B,0.84,46,[30,6],0.58,1445,950)

if __name__=='__main__':
    F=float(sys.argv[1]); out=sys.argv[2]
    lib.setF(F)
    names={1:'01_scopri-bonus-casino',2:'02_staryes-1050',3:'03_netwin-500',4:'04_pokerstars-310',5:'05_confronta-bonus-casino'}
    for i,fn in [(1,slide1),(2,slide2),(3,slide3),(4,slide4),(5,slide5)]:
        cv,info=fn()
        save(cv,f'{out}/{names[i]}.png'); print(i,cv.shape,info)
