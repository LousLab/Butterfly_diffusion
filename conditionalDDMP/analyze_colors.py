import os, csv
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

DATASET_NAME = 'huggan/smithsonian_butterflies_subset'
OUTPUT_DIR = '../hsv_color_analysis_output'
MIN_SATURATION, MIN_VALUE, MAX_VALUE = 50, 35, 245
COLOR_PRESENCE_THRESHOLD = 5.0
MIN_DDPM_SAMPLES = 50
COLORS = ['Red','Orange','Yellow','Green','Cyan','Blue','Purple','Brown']
DISPLAY_RGB = {'Red':(230,40,40),'Orange':(235,125,20),'Yellow':(235,205,40),'Green':(45,170,70),'Cyan':(30,190,200),'Blue':(45,90,210),'Purple':(145,65,170),'Brown':(125,75,40)}
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_masks(hsv):
    h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    usable=(s>=MIN_SATURATION)&(v>=MIN_VALUE)&(v<=MAX_VALUE)
    raw={
      'Red':usable&(((h<=8)|(h>=171))&(s>=90)&(v>=70)),
      'Orange':usable&((h>=9)&(h<=20)&(s>=125)&(v>=80)),
      'Yellow':usable&((h>=21)&(h<=38)&(s>=90)&(v>=90)),
      'Green':usable&((h>=39)&(h<=85)&(s>=80)&(v>=60)),
      'Cyan':usable&((h>=86)&(h<=100)&(s>=80)&(v>=60)),
      'Blue':usable&((h>=101)&(h<=130)&(s>=80)&(v>=50)),
      'Purple':usable&((h>=131)&(h<=170)&(s>=70)&(v>=50)),
      'Brown':usable&((h>=5)&(h<=30)&(s>=35)&(s<125)&(v>=35)&(v<=190))}
    assigned=np.zeros(h.shape,dtype=bool); out={}
    for name in COLORS:
        out[name]=raw[name]&~assigned
        assigned|=out[name]
    return out

def analyze(image):
    rgb=np.array(image.convert('RGB')); hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    masks=get_masks(hsv); total=int(sum(np.sum(m) for m in masks.values()))
    if total<50: return {},rgb,False
    return {n:100*np.sum(m)/total for n,m in masks.items()},rgb,True

def main():
    print(f'Loading {DATASET_NAME}...')
    ds=load_dataset(DATASET_NAME,split='train'); total=len(ds)
    counts=defaultdict(int); areas=defaultdict(list); samples=defaultdict(list); rows=[]
    for idx,item in enumerate(ds):
        props,rgb,ok=analyze(item['image'])
        if not ok: continue
        detected=[n for n,p in props.items() if p>=COLOR_PRESENCE_THRESHOLD]
        ranked=sorted(props.items(),key=lambda x:x[1],reverse=True)
        dom=ranked[0][0] if ranked[0][1]>=COLOR_PRESENCE_THRESHOLD else 'none'
        sec=ranked[1][0] if len(ranked)>1 and ranked[1][1]>=COLOR_PRESENCE_THRESHOLD else 'none'
        rows.append({'image_id':idx,'dominant_color':dom,'secondary_color':sec,'detected_colors':','.join(detected)})
        for n in detected:
            counts[n]+=1; areas[n].append(props[n]); samples[n].append((props[n],rgb,idx))
    csv_path=os.path.join(OUTPUT_DIR,'color_analysis.csv')
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','dominant_color','secondary_color','detected_colors']); w.writeheader(); w.writerows(rows)
    print('\n'+'='*90); print('CORRECTED HSV COLOR ANALYSIS'); print('='*90)
    print(f"{'Color':<10}{'Images':<10}{'% Dataset':<12}{'Avg Area %':<14}{'Max Area %':<14}{'DDPM'}")
    for n in COLORS:
        c=counts[n]; pd=100*c/total; a=np.mean(areas[n]) if areas[n] else 0; mx=np.max(areas[n]) if areas[n] else 0
        print(f'{n:<10}{c:<10}{pd:<12.1f}{a:<14.1f}{mx:<14.1f}{"YES" if c>=MIN_DDPM_SAMPLES else "NO"}')
    print('-'*90); print('Candidate classes:',', '.join(n for n in COLORS if counts[n]>=MIN_DDPM_SAMPLES) or 'None')
    fig,ax=plt.subplots(1,len(COLORS),figsize=(18,3))
    for i,n in enumerate(COLORS):
        sw=np.zeros((100,100,3),dtype=np.uint8); sw[:]=DISPLAY_RGB[n]; ax[i].imshow(sw); ax[i].set_title(f'{n}\n{counts[n]} images'); ax[i].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'color_distribution.png'),dpi=200); plt.close()
    for n in COLORS:
        if not samples[n]: continue
        sel=sorted(samples[n],reverse=True,key=lambda x:x[0])[:5]; fig,ax=plt.subplots(1,5,figsize=(15,3)); fig.suptitle(f'{n} samples')
        for i,a in enumerate(ax):
            if i<len(sel): pct,img,idx=sel[i]; a.imshow(img); a.set_title(f'{pct:.1f}%\nID {idx}')
            a.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,f'samples_{n.lower()}.png'),dpi=200); plt.close()
    print(f'\nSaved: {csv_path}\nVisuals: {OUTPUT_DIR}/')
if __name__=='__main__': main()
