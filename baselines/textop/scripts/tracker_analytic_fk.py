"""Read-only named-link URDF FK. No simulator, dynamics or visual-model substitution."""
import xml.etree.ElementTree as ET
import numpy as np

def multiply(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.concatenate([a[...,:1]*b[...,:1]-np.sum(a[...,1:]*b[...,1:],axis=-1,keepdims=True),
        a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])],axis=-1)

def rotate(q,v):
    qv=q[...,1:];t=2*np.cross(qv,v)
    return v+q[...,:1]*t+np.cross(qv,t)

def axis_quat(axis,angle):
    angle=np.asarray(angle,dtype=np.float64)
    return np.concatenate([np.cos(angle/2)[...,None],np.sin(angle/2)[...,None]*axis],axis=-1)

class TrackerFK:
    def __init__(self,urdf,joint_names,body_names):
        self.joint_names=list(joint_names);self.body_names=list(body_names)
        assert len(set(self.joint_names))==len(self.joint_names)==29
        tree=ET.parse(urdf).getroot();links={n.attrib['name'] for n in tree.findall('link')}
        children=set();pending=[];movable=[]
        for j in tree.findall('joint'):
            assert j.find('mimic') is None
            kind=j.attrib['type'];assert kind in ['fixed','revolute','continuous']
            name=j.attrib['name'];pa=j.find('parent').attrib['link'];ch=j.find('child').attrib['link']
            assert ch not in children;children.add(ch)
            o=j.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
            oq=np.array([1.,0,0,0])
            for i in [2,1,0]:oq=multiply(oq,axis_quat(np.eye(3)[i],rpy[i]))
            a=j.find('axis');axis=np.fromstring(a.get('xyz','1 0 0') if a is not None else '1 0 0',sep=' ')
            assert xyz.shape==rpy.shape==axis.shape==(3,) and np.linalg.norm(axis)>0
            axis/=np.linalg.norm(axis)
            idx=None if kind=='fixed' else self.joint_names.index(name)
            if idx is not None:movable.append(name)
            pending.append((pa,ch,xyz,oq,axis,idx))
        assert set(movable)==set(self.joint_names)
        roots=links-children;assert roots=={'pelvis'};self.root='pelvis';self.ordered=[];seen=roots.copy()
        while pending:
            ready=[x for x in pending if x[0] in seen];assert ready,'Disconnected or cyclic URDF'
            for x in ready:self.ordered.append(x);seen.add(x[1]);pending.remove(x)
        assert set(self.body_names)<=seen

    def forward(self,root_pos,root_quat,joints):
        p=np.asarray(root_pos,np.float64);q=np.asarray(root_quat,np.float64);j=np.asarray(joints,np.float64)
        assert p.shape[:-1]==q.shape[:-1]==j.shape[:-1] and p.shape[-1]==3 and q.shape[-1]==4 and j.shape[-1]==29
        assert np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(j).all()
        assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-4
        q=q/np.linalg.norm(q,axis=-1,keepdims=True);poses={self.root:(p,q)}
        for pa,ch,xyz,oq,axis,idx in self.ordered:
            pp,pq=poses[pa];cq=multiply(pq,oq)
            if idx is not None:cq=multiply(cq,axis_quat(axis,j[...,idx]))
            poses[ch]=(pp+rotate(pq,xyz),cq/np.linalg.norm(cq,axis=-1,keepdims=True))
        return np.stack([poses[n][0] for n in self.body_names],axis=-2),np.stack([poses[n][1] for n in self.body_names],axis=-2)

def reference_arrays(fk,qpos,input_names,dt=.02):
    """Last frame velocity is provisional; streaming consumers must exclude it."""
    q=np.asarray(qpos);assert len(q)>=3
    jp=q[:,7:][:,[input_names.index(n) for n in fk.joint_names]]
    bp,bq=fk.forward(q[:,:3],q[:,3:7],jp)
    prev=bq[:-1];nxt=bq[1:];inv=prev.copy();inv[...,1:]*=-1
    delta=multiply(nxt,inv);delta*=np.where(delta[...,:1]<0,-1.,1.)
    norm=np.linalg.norm(delta[...,1:],axis=-1)
    av=delta[...,1:]*(2*np.arctan2(norm,delta[...,0])/np.maximum(norm,1e-12))[...,None]/dt
    bav=np.concatenate([av[:1],(av[:-1]+av[1:])*.5,av[-1:]],axis=0)
    return dict(fps=np.array([round(1/dt)]),joint_pos=jp.astype(np.float32),joint_vel=np.gradient(jp,dt,axis=0).astype(np.float32),
        body_pos_w=bp.astype(np.float32),body_quat_w=bq.astype(np.float32),body_lin_vel_w=np.gradient(bp,dt,axis=0).astype(np.float32),body_ang_vel_w=bav.astype(np.float32))
