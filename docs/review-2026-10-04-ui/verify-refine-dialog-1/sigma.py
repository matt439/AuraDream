import math
def mu(w,h):
    seq=(w//16)*(h//16)
    m=(1.15-0.5)/(4096-256); b=0.5-m*256
    return seq*m+b
def sig(w,h,strength,steps=28,mu_override=None):
    t_start=int(max(steps-min(steps*strength,steps),0))
    ts=[1.0-i*(1.0-1.0/steps)/(steps-1) for i in range(steps)]
    t=ts[t_start]
    M=mu(w,h) if mu_override is None else mu_override
    return round(math.exp(M)/(math.exp(M)+(1/t-1)),3), round(M,2), steps-t_start
for name,(w,h) in {'1024->1536 1.5x':(1536,1536),'fox 1.5x':(1344,1728),'harbour 1.5x':(2496,1392),'harbour 1.25x':(2080,1168),'2048 2x':(2048,2048),'fox 2x':(1792,2304),'harbour 2x':(3328,1856),'4MP 1.25x':(2560,2560)}.items():
    for s in (0.05,0.1,0.15):
        print(name, s, sig(w,h,s))
print('mu1.15 at 0.2 2048', sig(2048,2048,0.2,mu_override=1.15))
