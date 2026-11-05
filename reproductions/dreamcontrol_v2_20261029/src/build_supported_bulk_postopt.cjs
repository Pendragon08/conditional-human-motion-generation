const f=require('fs'),p=__dirname+'/';let s=f.readFileSync(p+'stage3_supported_task_postopt_v2.py','utf8');
function rep(a,b){if(!s.includes(a))throw Error('missing '+a);s=s.replace(a,b)}
rep('def hand_cost(vals,q,r,position,orientation,hand_mask):','def hand_cost(vals,q,r,position,orientation,hand_mask,p_scale,r_scale):');
rep('hand_mask[:,None]*100.','hand_mask[:,None]*p_scale');rep('hand_mask[:,None]*30.','hand_mask[:,None]*r_scale');
rep('def stance_cost(vals,q,r):','def stance_cost(vals,q,r,p_scale,r_scale):');rep('(pose.translation()-foot_goal)*80.','(pose.translation()-foot_goal)*p_scale');rep('(foot_rotation.inverse()@pose.rotation()).log()*30.','(foot_rotation.inverse()@pose.rotation()).log()*r_scale');
rep('@partial(jax.jit,static_argnums=(0,))','@partial(jax.jit,static_argnums=(0,9))');
rep('def solve(self,target,goals,orientation,hand_mask,initial_q,initial_root):','def solve(self,target,goals,orientation,hand_mask,initial_q,initial_root,limit_weight,scales,refine):');
rep('fit,hand,stance,root,ground,table=self.costs','fit,hand,stance,root,ground,table=self.costs\n        weights=jnp.broadcast_to(scales[None],(180,4))');
rep('hand(q,r,goals,orientation,hand_mask),stance(q,r)','hand(q,r,goals,orientation,hand_mask,weights[:,0],weights[:,1]),stance(q,r,weights[:,2],weights[:,3])');
rep('jax.tree.map(lambda x:x[None],robot),q,1000.)','jax.tree.map(lambda x:x[None],robot),q,limit_weight)');
rep('solved=jaxls.LeastSquaresProblem(costs,[q,r]).analyze().solve(initial_vals=initial)',`problem=jaxls.LeastSquaresProblem(costs,[q,r]).analyze()
        if refine:
            solved=problem.solve(initial_vals=initial,linear_solver=jaxls.ConjugateGradientConfig(tolerance_max=1e-5),trust_region=jaxls.TrustRegionConfig(lambda_max=1e12),termination=jaxls.TerminationConfig(max_iterations=200,cost_tolerance=1e-9,parameter_tolerance=1e-9))
        else:solved=problem.solve(initial_vals=initial)`);
let a=s.indexOf('        raw_q,raw_root=self.solve('),b=s.indexOf('        raw_q=np.asarray(raw_q);raw_root=np.asarray(raw_root)',a);if(a<0||b<0)throw Error('stages');
s=s.slice(0,a)+`        t=jnp.array(target,dtype=jnp.float32);g=jnp.array(goals[:,20:22],dtype=jnp.float32);o=jnp.array(orientations,dtype=jnp.float32);hm=jnp.array(hand_mask)
        qa,ra=self.solve(t,g,o,hm,jnp.array(initial_q),jnp.array(initial_root),jnp.float32(1000.),jnp.array([100.,30.,80.,30.]),False)
        qa=np.asarray(qa);ra=np.asarray(ra);assert np.isfinite(qa).all() and np.isfinite(ra).all(),'nonfinite initial stage preserved by guard'
        stage=output/'initial_stage.npz'
        with stage.open('xb') as f:np.savez_compressed(f,q=qa,root=ra)
        iq=np.clip(qa.astype(float),self.low+1e-6,self.high-1e-6).astype(np.float32)
        initial_sha=sha(stage);initial_values_sha=hashlib.sha256(iq.tobytes()+ra.tobytes()).hexdigest()
        scales=[300.,30.,240.,30.] if m['task']=='box_push_pull' else [100.,30.,80.,30.]
        raw_q,raw_root=self.solve(t,g,o,hm,jnp.array(iq),jnp.array(ra),jnp.float32(10000.),jnp.array(scales),False)
        raw_q=np.asarray(raw_q);raw_root=np.asarray(raw_root)
        cq=np.clip(raw_q.astype(float),self.low+1e-6,self.high-1e-6).astype(np.float32)
        pp,rr=self.fk.forward(raw_root[:,4:],raw_root[:,[1,2,3,0]],cq,self.names,links=self.all_links,return_rotations=True)
        ee=np.linalg.norm(pp[:,self.wrist]-goals[:,20:22],axis=-1)[hand_mask.astype(bool)]
        aa=(Rotation.from_matrix(rr[:,self.wrist].reshape(-1,3,3)).inv()*Rotation.from_quat(orientations.reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(180,2)[hand_mask.astype(bool)]
        sole=[]
        for name,v in self.cpu_meshes.items():
            k=self.all_links.index(name)
            if k in self.feet:sole.append((np.einsum('tij,vj->tvi',rr[:,k],v)+pp[:,k,None])[:,:,2].min(axis=1))
        refined=bool(np.max(np.abs(cq-raw_q))>=.001 or ee.max()>=.03 or aa.max()>=.20 or np.stack(sole,1).max()>=.02)
        if refined:
            stage=output/'warm_stage.npz'
            with stage.open('xb') as f:np.savez_compressed(f,q=raw_q,root=raw_root)
            initial_sha=sha(stage);initial_values_sha=hashlib.sha256(cq.tobytes()+raw_root.tobytes()).hexdigest()
            scales=[300.,90.,480.,90.] if m['task']=='box_push_pull' else [100.,30.,80.,30.]
            raw_q,raw_root=self.solve(t,g,o,hm,jnp.array(cq),jnp.array(raw_root),jnp.float32(10000.),jnp.array(scales),True)
`+s.slice(b);
s=s.split("'initial_state_sha256':initial_sha").join("'initial_state_sha256':initial_sha,'initial_values_sha256':initial_values_sha,'initialization':'two-stage fitting from neutral; optional stricter numerical refinement','extra_refinement_used':refined");
f.writeFileSync(p+'stage3_supported_bulk_postopt_v1.py',s,{flag:'wx'});
