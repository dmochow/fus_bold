# -*- coding: utf-8 -*-
from lxml import etree
from io import BytesIO
import re
from pathlib import Path

SVGNS = "http://www.w3.org/2000/svg"
INK   = "http://www.inkscape.org/namespaces/inkscape"

def parse_style(s):
    d={}
    if s:
        for kv in s.split(';'):
            if ':' in kv:
                k,v = kv.split(':',1); d[k.strip()]=v.strip()
    return d
def style_str(d): return ';'.join(f"{k}:{v}" for k,v in d.items())

def get_font_size(el):
    fs = el.get("font-size")
    if fs and re.search(r'[\d.]+',fs): return float(re.search(r'[\d.]+',fs).group())
    st = parse_style(el.get("style",""))
    if "font-size" in st and re.search(r'[\d.]+',st["font-size"]):
        return float(re.search(r'[\d.]+',st["font-size"]).group())
    return None

def set_font_size(el,px):
    st = parse_style(el.get("style","")); st["font-size"]=f"{px}px"
    el.set("style",style_str(st)); el.set("font-size",f"{px}px")

def find_panel_group(root,label_regex):
    rx=re.compile(label_regex)
    for t in root.findall(f".//{{{SVGNS}}}text"):
        if rx.match(''.join(t.itertext()).strip()):
            g=t.getparent(); last=None; hops=0
            while g is not None and hops<6:
                if g.tag.endswith('g'): last=g
                g=g.getparent(); hops+=1
            return last
    return None

def double_fonts(g,log):
    n=0
    for t in g.findall(f".//{{{SVGNS}}}text"):
        fs=get_font_size(t)
        if fs: set_font_size(t,fs*2); n+=1
    log.append(f"  Doubled font sizes in {n} text nodes.")

def panel_bbox(g):
    best=None; area=-1
    for r in g.findall(f".//{{{SVGNS}}}rect"):
        try:
            x=float(r.get('x','0')); y=float(r.get('y','0'))
            w=float(r.get('width','0')); h=float(r.get('height','0'))
            a=w*h
            if a>area: best=(x,y,w,h); area=a
        except: pass
    return best

def scale_rect(r,sx,sy):
    try:
        x=float(r.get('x','0')); y=float(r.get('y','0'))
        w=float(r.get('width','0')); h=float(r.get('height','0'))
    except: return
    cx=x+w/2; cy=y+h/2
    nw=w*sx; nh=h*sy
    r.set('x',str(cx-nw/2)); r.set('y',str(cy-nh/2))
    r.set('width',str(nw));   r.set('height',str(nh))

def grow_timeline(g,log,grow=1.35,sono_w=1.5,sono_h=1.35):
    # Enlarge small “sonication” boxes (heuristic)
    n=0
    for r in g.findall(f".//{{{SVGNS}}}rect"):
        try:
            w=float(r.get('width','0')); h=float(r.get('height','0'))
            if h<50 and w<200:
                scale_rect(r,sono_w,sono_h); n+=1
        except: pass
    log.append(f"  Enlarged {n} small boxes (likely sonications).")

    # Try to find a group labelled like “time*” and scale it
    tl=None
    for cand in g.findall(f".//{{{SVGNS}}}g"):
        id_=(cand.get('id') or '').lower()
        lab=(cand.get(f'{{{INK}}}label') or '').lower()
        if 'time' in id_ or 'time' in lab:
            tl=cand; break
    if tl is None:
        # fallback: choose group with most rects+lines
        best=None; score=-1
        for cand in g.findall(f".//{{{SVGNS}}}g"):
            s=len(cand.findall(f'.//{{{SVGNS}}}rect'))+len(cand.findall(f'.//{{{SVGNS}}}line'))
            if s>score: score=s; best=cand
        tl=best
    if tl is not None:
        tl.set('transform', (tl.get('transform','')+f" scale({grow})").strip())
        log.append(f"  Scaled timeline group ×{grow:.2f}.")
    else:
        log.append("  Timeline group not found (skipped).")

def tighten_panel(g,log,margin=0.05):
    bb=panel_bbox(g)
    if not bb: 
        log.append("  No panel background rect found; skip tighten.")
        return
    px,py,pw,ph=bb
    xs=[]; ys=[]; xe=[]; ye=[]
    for r in g.findall(f".//{{{SVGNS}}}rect"):
        try:
            x=float(r.get('x','0')); y=float(r.get('y','0'))
            w=float(r.get('width','0')); h=float(r.get('height','0'))
            xs+=[x]; ys+=[y]; xe+=[x+w]; ye+=[y+h]
        except: pass
    for t in g.findall(f".//{{{SVGNS}}}text"):
        try:
            x=float(t.get('x','0')); y=float(t.get('y','0'))
            xs+=[x]; ys+=[y]; xe+=[x]; ye+=[y]
        except: pass
    if not xs: 
        log.append("  Could not compute content bounds.")
        return
    cx0,cy0,cx1,cy1=min(xs),min(ys),max(xe),max(ye)
    cw,ch=cx1-cx0,cy1-cy0
    tx=px+pw*margin; ty=py+ph*margin
    tw=pw*(1-2*margin); th=ph*(1-2*margin)
    sx=tw/cw if cw>0 else 1; sy=th/ch if ch>0 else 1; s=min(sx,sy)
    # Wrap content and transform
    wrapper=etree.Element(f"{{{SVGNS}}}g")
    for child in list(g): wrapper.append(child); g.remove(child)
    g.append(wrapper)
    wrapper.set('transform',f"translate({tx},{ty}) scale({s}) translate({-cx0},{-cy0})")
    log.append(f"  Tightened content (scale ×{s:.2f}, ~{int(margin*100)}% margin).")

def modify_svg(inp,outp,grow=1.35,sono=(1.5,1.35),tighten=True):
    svg=etree.parse(BytesIO(Path(inp).read_bytes())).getroot()
    log=[]
    panels={'d':find_panel_group(svg,r'^\(?[dD]\)?'),
            'e':find_panel_group(svg,r'^\(?[eE]\)?'),
            'f':find_panel_group(svg,r'^\(?[fF]\)?')}
    for k,g in panels.items():
        if g is None: 
            log.append(f"[panel {k}] not found."); 
            continue
        log.append(f"[panel {k}]")
        double_fonts(g,log)
        if k=='d': grow_timeline(g,log,grow,sono[0],sono[1])
        if tighten: tighten_panel(g,log)
    Path(outp).write_bytes(etree.tostring(svg,pretty_print=True,xml_declaration=True,encoding='UTF-8'))
    return "\n".join(log)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--grow',type=float,default=1.35)
    ap.add_argument('--sono-w',type=float,default=1.5)
    ap.add_argument('--sono-h',type=float,default=1.35)
    ap.add_argument('--no-tighten',action='store_true')
    a=ap.parse_args()
    print(modify_svg(a.input,a.output,grow=a.grow,sono=(a.sono_w,a.sono_h),tighten=not a.no_tighten))