#!/usr/bin/env python3
"""Full-night PSG/ear-EEG spectra and PSG-defined N2 sigma-window selection."""
from __future__ import annotations
import argparse, json
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy import signal

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("psg",type=Path); p.add_argument("ear",type=Path); p.add_argument("stages",type=Path)
    p.add_argument("--out-dir",type=Path,default=Path("spindle_results")); p.add_argument("--stage-col",type=int,default=2)
    p.add_argument("--stage-sheet",default="0"); p.add_argument("--stage-offset",type=float,default=0)
    p.add_argument("--target-fs",type=float,default=200); p.add_argument("--notch",type=float,default=None)
    p.add_argument("--top-k",type=int,default=3); p.add_argument("--window-min",type=float,default=2)
    p.add_argument("--step-sec",type=float,default=30); p.add_argument("--export-full-csv",action="store_true")
    return p.parse_args()

def load_eeg(path):
    a=np.loadtxt(path,ndmin=2)
    if a.shape[1]<2: raise ValueError(f"{path}: need time(s), EEG(uV)")
    t,x=a[:,0].astype(float),a[:,1].astype(float)
    if len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)): raise ValueError(f"{path}: nonfinite/short data")
    dt=np.diff(t); med=float(np.median(dt))
    if np.any(dt<=0) or np.max(np.abs(dt-med))>max(.05*med,1e-6): raise ValueError(f"{path}: timestamp gap/irregularity")
    return t,x,1/med

def align(t1,x1,fs1,t2,x2,fs2):
    if not np.isclose(fs1,fs2,rtol=1e-4,atol=1e-6): raise ValueError(f"Sampling rates differ: {fs1} vs {fs2}")
    n=min(len(t1),len(t2)); a=t1[:n]-t1[0]; b=t2[:n]-t2[0]
    if np.max(np.abs(a-b))>max(.1/fs1,1e-6): raise ValueError("PSG and ear-EEG are not sample-aligned")
    return a,x1[:n],x2[:n],(fs1+fs2)/2

def resample_pair(t,x1,x2,fs,target):
    if target is None or np.isclose(fs,target,rtol=1e-8): return t,x1,x2,fs
    q=Fraction(float(target/fs)).limit_denominator(100000)
    y1=signal.resample_poly(x1,q.numerator,q.denominator); y2=signal.resample_poly(x2,q.numerator,q.denominator)
    n=min(len(y1),len(y2)); return np.arange(n)/target,y1[:n],y2[:n],target

def preprocess(x,fs,notch):
    x=signal.detrend(x,type="constant")
    if notch is not None:
        if not 0<notch<fs/2: raise ValueError("Invalid notch")
        x=signal.sosfiltfilt(signal.tf2sos(*signal.iirnotch(notch,Q=30,fs=fs)),x)
    return signal.sosfiltfilt(signal.butter(4,(.5,30),btype="bandpass",fs=fs,output="sos"),x)

def load_stages(path,col,sheet):
    sheet=int(sheet) if str(sheet).isdigit() else sheet
    d=pd.read_excel(path,sheet_name=sheet) if path.suffix.lower() in (".xlsx",".xls") else pd.read_csv(path)
    if col>=d.shape[1]: raise ValueError("Stage column out of range")
    s=d.iloc[:,col].astype("string").str.strip().str.upper().replace({"WAKE":"W","REM":"R","SLEEP-WAKE":"W"})
    return s.fillna("UNKNOWN").to_numpy()

def stage_mask_at_times(stages,times,offset,keep=("N2",)):
    index=np.floor(times-offset).astype(int); valid=(index>=0)&(index<len(stages)); out=np.zeros(len(times),bool)
    out[valid]=np.isin(stages[index[valid]],keep); return out

def windows(x,fs,win_s,hop_s):
    n=int(round(win_s*fs)); hop=int(round(hop_s*fs)); starts=np.arange(0,len(x)-n+1,hop)
    if n<16 or hop<1 or not len(starts): raise ValueError("Invalid/too-short windowing")
    return np.stack([x[i:i+n] for i in starts]),starts

def basic_qc(w):
    rms=np.sqrt(np.mean(w*w,axis=1)); ptp=np.ptp(w,axis=1); ok=np.all(np.isfinite(w),axis=1)&(rms>0)&(ptp>0)
    if not ok.any(): return ok
    for z in (np.log(rms+1e-30),np.log(ptp+1e-30)):
        med=np.median(z[ok]); mad=1.4826*np.median(np.abs(z[ok]-med))
        if mad>0: ok &= np.abs(z-med)<=5*mad
    return ok

def mne_mt(x,fs,win_s,hop_s,fmin,fmax,bw,batch=256):
    w,starts=windows(x,fs,win_s,hop_s); blocks=[]; freqs=None
    for i in range(0,len(w),batch):
        p,f=psd_array_multitaper(w[i:i+batch]*1e-6,sfreq=fs,fmin=fmin,fmax=fmax,bandwidth=bw,
                                 adaptive=True,low_bias=True,normalization="full",verbose=False)
        if freqs is None: freqs=f
        elif not np.allclose(freqs,f): raise RuntimeError("Frequency grid changed")
        blocks.append(p)
    return freqs,(starts+w.shape[1]/2)/fs,np.concatenate(blocks).T,basic_qc(w)

def hann_psd(x,fs,win_s,hop_s,fmin,fmax):
    noverlap=int(round((win_s-hop_s)*fs))
    f,t,p=signal.spectrogram(x*1e-6,fs=fs,window="hann",nperseg=int(round(win_s*fs)),noverlap=noverlap,
                             detrend=False,scaling="density",mode="psd")
    k=(f>=fmin)&(f<=fmax); return f[k],t,p[k]

def sigma_events(psg,fs,n2_sample_mask,band=(13.5,16),min_s=.5,max_s=2.5):
    y=signal.sosfiltfilt(signal.butter(4,band,btype="bandpass",fs=fs,output="sos"),psg)
    env=np.abs(signal.hilbert(y)); size=max(1,int(.2*fs)); env=np.convolve(env,np.ones(size)/size,mode="same")
    base=env[n2_sample_mask]
    if len(base)<int(60*fs): raise ValueError("Less than 60 s of N2 available")
    med=np.median(base); scale=1.4826*np.median(np.abs(base-med)); scale=max(scale,np.finfo(float).eps)
    active=(env>=med+2*scale)&(env<=med+10*scale)&n2_sample_mask
    edges=np.diff(np.r_[False,active,False].astype(int)); starts=np.flatnonzero(edges==1); stops=np.flatnonzero(edges==-1)
    keep=((stops-starts)/fs>=min_s)&((stops-starts)/fs<=max_s)
    return [(a/fs,b/fs) for a,b in zip(starts[keep],stops[keep])],env,med+2*scale

def second_qc(psg,fs):
    n=int(fs); count=len(psg)//n; w=psg[:count*n].reshape(count,n); return basic_qc(w)

def select_windows(events,stage_1hz,qc_1hz,total_s,win_s,step_s,top_k,min_stage=.9):
    candidates=[]
    for start in np.arange(0,total_s-win_s+1,step_s):
        end=start+win_s; i0,i1=int(np.floor(start)),int(np.ceil(end)); stage=stage_1hz[i0:i1]; qc=qc_1hz[i0:i1]
        if len(stage)<win_s*.95 or stage.mean()<min_stage or qc.mean()<.9: continue
        count=sum(start<=a and b<=end for a,b in events); candidates.append((count/(stage.sum()/60),start,end,count,stage.mean(),qc.mean()))
    chosen=[]
    for row in sorted(candidates,reverse=True):
        if all(row[2]<=x[1] or row[1]>=x[2] for x in chosen): chosen.append(row)
        if len(chosen)>=top_k: break
    return chosen

def save_pair_png(path,f,t,a,b,title,band=(.5,30)):
    k=(f>=band[0])&(f<=band[1]); eps=np.finfo(float).tiny; A=10*np.log10(a[k]+eps); B=10*np.log10(b[k]+eps)
    finite=np.r_[A[np.isfinite(A)],B[np.isfinite(B)]]
    if finite.size==0: raise ValueError("No finite spectrogram values to plot")
    lo,hi=np.percentile(finite,[5,95])
    fig,ax=plt.subplots(2,1,figsize=(14,7),sharex=True,constrained_layout=True)
    for axis,z,name in zip(ax,(A,B),("PSG","Ear-EEG")):
        im=axis.pcolormesh(t,f[k],z,shading="auto",vmin=lo,vmax=hi,cmap="viridis"); axis.set(title=name,ylabel="Frequency (Hz)")
        fig.colorbar(im,ax=axis,label="10 log10(PSD / 1 V² Hz⁻¹)")
    ax[-1].set_xlabel("Time (s)"); fig.suptitle(title); fig.savefig(path,dpi=300); plt.close(fig)

def main():
    a=arguments(); a.out_dir.mkdir(parents=True,exist_ok=True)
    t1,x1,fs1=load_eeg(a.psg); t2,x2,fs2=load_eeg(a.ear); t,x1,x2,fs=align(t1,x1,fs1,t2,x2,fs2)
    t,x1,x2,fs=resample_pair(t,x1,x2,fs,a.target_fs); x1=preprocess(x1,fs,a.notch); x2=preprocess(x2,fs,a.notch)
    stages=load_stages(a.stages,a.stage_col,a.stage_sheet)
    sample_n2=stage_mask_at_times(stages,t,a.stage_offset); sec_t=np.arange(int(np.floor(t[-1]))+1)
    stage_1hz=stage_mask_at_times(stages,sec_t,a.stage_offset); qc_1hz=second_qc(x1,fs)[:len(sec_t)]; stage_1hz=stage_1hz[:len(qc_1hz)]

    f,tm,p1,q1=mne_mt(x1,fs,4,2,.5,30,2); f2,tm2,p2,q2=mne_mt(x2,fs,4,2,.5,30,2)
    if not np.allclose(f,f2) or not np.allclose(tm,tm2): raise RuntimeError("Spectrogram grids differ")
    valid=q1&q2; p1[:,~valid]=np.nan; p2[:,~valid]=np.nan
    np.savez_compressed(a.out_dir/"fullnight_MNE_multitaper.npz",frequencies_Hz=f,times_s=tm,
                        PSG_psd_V2_per_Hz=p1,EAR_psd_V2_per_Hz=p2,valid_window=valid)
    save_pair_png(a.out_dir/"fullnight_MNE_multitaper.png",f,tm,p1,p2,"Full-night MNE multitaper")
    fsn,tsn,s1=hann_psd(x1,fs,4,2,.5,30); _,_,s2=hann_psd(x2,fs,4,2,.5,30)
    if len(tsn)!=len(valid) or not np.allclose(tsn,tm): raise RuntimeError("Hann and multitaper grids differ")
    s1[:,~valid]=np.nan; s2[:,~valid]=np.nan
    np.savez_compressed(a.out_dir/"fullnight_Hann.npz",frequencies_Hz=fsn,times_s=tsn,PSG_psd_V2_per_Hz=s1,EAR_psd_V2_per_Hz=s2)
    if a.export_full_csv:
        pd.DataFrame(p1,index=f,columns=tm).to_csv(a.out_dir/"PSG_fullnight_MT.csv")
        pd.DataFrame(p2,index=f,columns=tm).to_csv(a.out_dir/"EAR_fullnight_MT.csv")

    events,_,threshold=sigma_events(x1,fs,sample_n2); win_s=a.window_min*60
    chosen=select_windows(events,stage_1hz,qc_1hz,len(stage_1hz),win_s,a.step_sec,a.top_k)
    pd.DataFrame(chosen,columns=("events_per_N2_min","start_s","end_s","event_count","N2_fraction","PSG_QC_fraction")).to_csv(a.out_dir/"PSG_selected_windows.csv",index=False)
    window_dir=a.out_dir/"PSG_selected_windows"; window_dir.mkdir(exist_ok=True)
    for rank,row in enumerate(chosen,1):
        _,start,end,_,_,_=row; i0,i1=int(round(start*fs)),int(round(end*fs))
        fw,tw,z1,v1=mne_mt(x1[i0:i1],fs,4,.2,.5,30,1); fw2,tw2,z2,v2=mne_mt(x2[i0:i1],fs,4,.2,.5,30,1)
        if not np.allclose(fw,fw2) or not np.allclose(tw,tw2): raise RuntimeError("Window grids differ")
        good=v1&v2; z1[:,~good]=np.nan; z2[:,~good]=np.nan; tw=tw+start
        np.savez_compressed(window_dir/f"window_{rank}.npz",frequencies_Hz=fw,times_s=tw,PSG=z1,EAR=z2,valid_window=good)
        save_pair_png(window_dir/f"window_{rank}.png",fw,tw,z1,z2,f"PSG-defined N2 sigma window {rank}",(10,20))
    settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    settings.update(input_fs=float(fs1),analysis_fs=float(fs),sigma_band_Hz=[13.5,16],sigma_threshold_uV=float(threshold),
                    sigma_definition="PSG only; robust envelope 2-10 MAD; duration 0.5-2.5 s")
    (a.out_dir/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    print(f"Saved to {a.out_dir.resolve()}; PSG-defined sigma events={len(events)}, selected windows={len(chosen)}")

if __name__=="__main__": main()
