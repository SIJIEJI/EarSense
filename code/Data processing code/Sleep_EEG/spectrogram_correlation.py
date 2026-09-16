#!/usr/bin/env python3
"""Strict comparison of two already-computed EEG spectrogram CSV files."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

BANDS={"delta":(.5,4),"theta":(4,8),"alpha":(8,12),"sigma":(12,16),"beta":(16,30)}

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("psg_csv",type=Path); p.add_argument("ear_csv",type=Path)
    p.add_argument("--out-dir",type=Path,default=Path("spectrogram_comparison"))
    p.add_argument("--smooth-sec",type=float,default=120)
    p.add_argument("--time-atol",type=float,default=1e-6)
    p.add_argument("--freq-atol",type=float,default=1e-9)
    return p.parse_args()

def parse_time(value):
    if value is None: return np.nan
    text=str(value).strip()
    if not text or text.lower() in ("nan","#qnan","qnan") or set(text)=={"#"}: return np.nan
    try: return float(text)
    except ValueError: pass
    parts=text.split(":")
    try:
        if len(parts)==2: return 60*float(parts[0])+float(parts[1])
        if len(parts)==3: return 3600*float(parts[0])+60*float(parts[1])+float(parts[2])
    except ValueError: pass
    return np.nan

def parse_spectrogram(path):
    raw=pd.read_csv(path,header=None,dtype=str,keep_default_na=False)
    freq_cols=[]; freqs=[]
    for column,value in enumerate(raw.iloc[0,1:],start=1):
        try:
            number=float(str(value).strip())
            if np.isfinite(number): freq_cols.append(column); freqs.append(number)
        except ValueError: continue
    if len(freqs)<2: raise ValueError(f"{path}: fewer than two valid frequencies in first row")
    times=[]; rows=[]
    for row in range(1,len(raw)):
        time=parse_time(raw.iat[row,0])
        if not np.isfinite(time): continue
        values=pd.to_numeric(raw.iloc[row,freq_cols],errors="coerce").to_numpy(float)
        if np.isfinite(values).any(): times.append(time); rows.append(values)
    if len(times)<2: raise ValueError(f"{path}: fewer than two valid time rows")
    f=np.asarray(freqs); t=np.asarray(times); s=np.asarray(rows).T
    fi=np.argsort(f); ti=np.argsort(t); f,t,s=f[fi],t[ti],s[np.ix_(fi,ti)]
    if np.any(np.diff(f)<=0) or np.any(np.diff(t)<=0): raise ValueError(f"{path}: duplicate/non-increasing axes")
    if np.any(np.isfinite(s)&(s<0)): raise ValueError(f"{path}: PSD contains negative values")
    return f,t,s

def require_common_grid(f1,t1,s1,f2,t2,s2,f_atol,t_atol):
    if len(f1)!=len(f2) or not np.allclose(f1,f2,rtol=0,atol=f_atol):
        raise ValueError("Frequency axes differ; regenerate both spectrograms with identical settings")
    if len(t1)!=len(t2) or not np.allclose(t1,t2,rtol=0,atol=t_atol):
        raise ValueError("Time axes differ; regenerate/alignment-correct upstream rather than interpolating here")
    if s1.shape!=s2.shape: raise ValueError("Spectrogram shapes differ")
    return f1,t1,s1,s2

def safe_log_db(s):
    out=np.full_like(s,np.nan,dtype=float); valid=np.isfinite(s)&(s>0); out[valid]=10*np.log10(s[valid]); return out

def correlation(a,b,min_n=10):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<min_n: return np.nan
    x,y=a[ok],b[ok]
    if np.std(x)==0 or np.std(y)==0: return np.nan
    return float(np.corrcoef(x,y)[0,1])

def integrate_band(s,f,limits):
    k=(f>=limits[0])&(f<limits[1])
    if k.sum()<2: return np.full(s.shape[1],np.nan)
    valid=np.sum(np.isfinite(s[k]),axis=0)>=2
    out=np.full(s.shape[1],np.nan)
    out[valid]=np.trapezoid(s[k][:,valid],x=f[k],axis=0)
    return out

def relative_spectrum(s,f,total=(.5,30)):
    total_power=integrate_band(s,f,total); df=np.gradient(f)
    density=s*df[:,None]; denom=np.nansum(density,axis=0)
    out=np.full_like(s,np.nan); valid=np.isfinite(denom)&(denom>0)
    out[:,valid]=density[:,valid]/denom[valid]; return out,total_power

def spectral_zscore(log_s):
    mean=np.nanmean(log_s,axis=0,keepdims=True); sd=np.nanstd(log_s,axis=0,keepdims=True)
    out=np.full_like(log_s,np.nan); valid=(sd[0]>0)&np.isfinite(sd[0]); out[:,valid]=(log_s[:,valid]-mean[:,valid])/sd[:,valid]
    return out

def centered_smooth(x,bins):
    if bins<=1: return np.asarray(x,float)
    return pd.Series(x).rolling(bins,center=True,min_periods=bins).mean().to_numpy()

def representation_summary(name,a,b,f):
    row={"representation":name,"overall_flattened_r_descriptive":correlation(a.ravel(),b.ravel())}
    spectral=np.array([correlation(a[:,i],b[:,i],5) for i in range(a.shape[1])])
    temporal=np.array([correlation(a[k],b[k]) for k in range(a.shape[0])])
    row.update(spectral_shape_r_mean=float(np.nanmean(spectral)),spectral_shape_r_median=float(np.nanmedian(spectral)),
               temporal_r_by_frequency_mean=float(np.nanmean(temporal)),temporal_r_by_frequency_median=float(np.nanmedian(temporal)))
    for band,limits in BANDS.items():
        k=(f>=limits[0])&(f<limits[1]); row[f"flattened_r_{band}_descriptive"]=correlation(a[k].ravel(),b[k].ravel())
    return row,spectral,temporal

def main():
    a=arguments(); a.out_dir.mkdir(parents=True,exist_ok=True)
    f1,t1,s1=parse_spectrogram(a.psg_csv); f2,t2,s2=parse_spectrogram(a.ear_csv)
    f,t,s1,s2=require_common_grid(f1,t1,s1,f2,t2,s2,a.freq_atol,a.time_atol)
    analysis_band=(f>=.5)&(f<=30)
    if analysis_band.sum()<2: raise ValueError("Need at least two frequency bins within 0.5-30 Hz")
    f,s1,s2=f[analysis_band],s1[analysis_band],s2[analysis_band]
    if len(t)<2: raise ValueError("Need at least two time bins")
    hop=float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t)-hop))>max(.05*hop,a.time_atol): raise ValueError("Time grid is irregular")
    smooth_bins=max(1,int(round(a.smooth_sec/hop)))
    log1,log2=safe_log_db(s1),safe_log_db(s2)
    rel1,_=relative_spectrum(s1,f); rel2,_=relative_spectrum(s2,f)
    z1,z2=spectral_zscore(log1),spectral_zscore(log2)
    reps={"linear_PSD":(s1,s2),"log10_PSD_dB":(log1,log2),
          "relative_spectral_power":(rel1,rel2),"zscored_log_spectral_shape":(z1,z2)}
    summaries=[]; by_time={"time_s":t}; by_freq={"frequency_Hz":f}
    for name,(x,y) in reps.items():
        row,rt,rf=representation_summary(name,x,y,f); summaries.append(row)
        by_time[f"spectral_shape_r_{name}"]=rt; by_freq[f"temporal_r_{name}"]=rf
    pd.DataFrame(summaries).to_csv(a.out_dir/"representation_correlation_summary.csv",index=False)
    pd.DataFrame(by_time).to_csv(a.out_dir/"spectral_shape_correlation_by_time.csv",index=False)
    pd.DataFrame(by_freq).to_csv(a.out_dir/"temporal_correlation_by_frequency.csv",index=False)

    total1=integrate_band(s1,f,(.5,30)); total2=integrate_band(s2,f,(.5,30)); traces={"time_s":t}
    band_rows=[]
    for name,limits in {**BANDS,"total":(.5,30)}.items():
        p1=total1 if name=="total" else integrate_band(s1,f,limits); p2=total2 if name=="total" else integrate_band(s2,f,limits)
        abs1,abs2=safe_log_db(p1),safe_log_db(p2)
        traces[f"PSG_{name}_absolute_dB_raw"]=abs1; traces[f"EAR_{name}_absolute_dB_raw"]=abs2
        traces[f"PSG_{name}_absolute_dB_smoothed"]=centered_smooth(abs1,smooth_bins)
        traces[f"EAR_{name}_absolute_dB_smoothed"]=centered_smooth(abs2,smooth_bins)
        row={"band":name,"absolute_dB_r_raw":correlation(abs1,abs2),
             "absolute_dB_r_smoothed":correlation(centered_smooth(abs1,smooth_bins),centered_smooth(abs2,smooth_bins))}
        if name!="total":
            r1,r2=p1/total1,p2/total2; rd1,rd2=safe_log_db(r1),safe_log_db(r2)
            traces[f"PSG_{name}_relative_dB_raw"]=rd1; traces[f"EAR_{name}_relative_dB_raw"]=rd2
            traces[f"PSG_{name}_relative_dB_smoothed"]=centered_smooth(rd1,smooth_bins)
            traces[f"EAR_{name}_relative_dB_smoothed"]=centered_smooth(rd2,smooth_bins)
            row.update(relative_dB_r_raw=correlation(rd1,rd2),
                       relative_dB_r_smoothed=correlation(centered_smooth(rd1,smooth_bins),centered_smooth(rd2,smooth_bins)))
        band_rows.append(row)
    pd.DataFrame(traces).to_csv(a.out_dir/"bandpower_timecourses.csv",index=False)
    pd.DataFrame(band_rows).to_csv(a.out_dir/"bandpower_correlation_summary.csv",index=False)
    settings={"psg_csv":str(a.psg_csv),"ear_csv":str(a.ear_csv),"smooth_seconds":a.smooth_sec,
              "smooth_bins":smooth_bins,"hop_seconds":hop,"frequency_bins":len(f),"time_bins":len(t),
              "note":"All flattened correlations are descriptive; no bin-level inferential p-values."}
    np.savez_compressed(a.out_dir/"spectrogram_comparison.npz",frequencies_Hz=f,times_s=t,PSG_PSD=s1,EAR_PSD=s2,
                        PSG_log_dB=log1,EAR_log_dB=log2,settings_json=json.dumps(settings))
    (a.out_dir/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    print(f"Saved to {a.out_dir.resolve()}")

if __name__=="__main__": main()
